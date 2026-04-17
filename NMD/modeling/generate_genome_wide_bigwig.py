"""Generate genome-wide NMD predictions as BED and bigWig for MANE Select transcripts.

For each MANE Select transcript, NMD efficiency is predicted at every codon
position with all three stop codons (TAG, TAA, TGA).  The mean prediction
across stop codons is assigned to every nucleotide in that codon, producing
full CDS coverage.  Output is a bedGraph-derived bigWig and a matching BED.
"""

from time import sleep

import numpy as np
import pandas as pd
import subprocess
import shutil
import tempfile
import torch
import typer
import urllib.request
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from genome_kit import Genome

from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.config import MODELS_DIR, GENCODE_VERSION, PROJ_ROOT, TABLES_DIR
from NMD.data.transcripts import create_six_track_encoding
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.utils import load_model, collate_fn

app = typer.Typer()

# ── UCSC tool paths (may live outside the active conda env) ──────────────────
_BEDGRAPH_TO_BIGWIG = shutil.which("bedGraphToBigWig") or "/home/mveiner/anaconda3/bin/bedGraphToBigWig"

# ── Stop-codon one-hot encodings ─────────────────────────────────────────────
_STOP_OHE = {
    "TAG": np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]),
    "TAA": np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]]),
    "TGA": np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]),
}

_STOP_CODONS = {"TAG", "TAA", "TGA"}
_INDEX_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}
_NUC_OHE = {
    "A": np.array([1, 0, 0, 0]),
    "C": np.array([0, 1, 0, 0]),
    "G": np.array([0, 0, 1, 0]),
    "T": np.array([0, 0, 0, 1]),
}


# ── Helper functions ─────────────────────────────────────────────────────────

def _setup_model(config: TrainerConfig):
    """Setup model and return (model, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NMDetectiveAI(
        hidden_dims=config.dnn_hidden_dims,
        dropout=config.dnn_dropout,
        random_init=config.random_init,
        use_mlm=config.Orthrus_MLM,
        activation_function=config.activation_function,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    return model, device


def _predict_sequences_batched(model, sequences, device, batch_size=16):
    """Run batched model inference on a list of encoded sequences."""
    if not sequences:
        return []
    dummy_df = pd.DataFrame({"y": [0.0] * len(sequences)})
    dataset = SequenceDataset(dummy_df, sequences, label_col="y")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    predictions = []
    with torch.no_grad():
        for batch_seqs, batch_lens, _ in loader:
            batch_seqs = batch_seqs.to(device)
            batch_lens = batch_lens.to(device)
            preds = model(batch_seqs, batch_lens).squeeze(-1)
            if preds.dim() == 0:
                predictions.append(float(preds.cpu()))
            else:
                predictions.extend(preds.cpu().tolist())
    return predictions


def _cds_to_genomic_positions(transcript):
    """Map each CDS nucleotide (in 5'→3' mRNA order) to its 0-based genomic coordinate."""
    strand = transcript.strand
    positions = []
    for cds in transcript.cdss:
        if strand == "+":
            positions.extend(range(cds.start, cds.end))
        else:
            positions.extend(range(cds.end - 1, cds.start - 1, -1))
    return positions


def _get_mane_select_transcripts(genome):
    """Return a list of MANE Select transcripts for *genome*.

    Uses genome_kit's built-in MANE annotation when available (e.g. gencode
    v41, v46), otherwise downloads the mapping from NCBI and resolves
    transcripts manually.

    Returns:
        list of genome_kit Transcript objects.
    """
    # Try built-in MANE first
    try:
        transcripts = genome.mane_select_transcripts()
        # Filter to coding transcripts only
        transcripts = [t for t in transcripts if t.cdss and len(t.cdss) > 0]
        logger.info(
            f"Using built-in MANE Select for {genome.config}: "
            f"{len(transcripts)} coding transcripts"
        )
        return transcripts
    except Exception:
        pass

    # Fallback: download from NCBI and resolve against this genome version
    logger.info("Built-in MANE not available, downloading from NCBI…")
    url = (
        "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/"
        "MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz"
    )
    mane = pd.read_csv(url, sep="\t", compression="gzip")
    mane_select = mane[mane["MANE_status"] == "MANE Select"]
    logger.info(f"Downloaded {len(mane_select)} MANE Select entries from NCBI")

    genes_by_name = {
        g.name: g for g in genome.genes
        if g.name and g.type == "protein_coding"
    }
    genes_by_ensg = {
        g.id.split(".")[0]: g for g in genome.genes
        if g.type == "protein_coding"
    }

    transcripts = []
    skipped = 0
    for _, row in mane_select.iterrows():
        gene_name = row["symbol"]
        ensg_base = row["Ensembl_Gene"].split(".")[0]
        tid_base = row["Ensembl_nuc"].split(".")[0]

        gene = genes_by_name.get(gene_name) or genes_by_ensg.get(ensg_base)
        if gene is None:
            logger.debug(f"Gene {gene_name} ({ensg_base}) not found")
            skipped += 1
            continue

        # Find the MANE transcript
        transcript = None
        for tr in gene.transcripts:
            if tr.id.split(".")[0] == tid_base and tr.cdss and len(tr.cdss) > 0:
                transcript = tr
                break

        # Fallback: longest coding transcript
        if transcript is None:
            coding = [tr for tr in gene.transcripts if tr.cdss]
            if coding:
                transcript = max(coding, key=lambda t: sum(len(c) for c in t.cdss))
                logger.debug(
                    f"MANE transcript {tid_base} not found for {gene_name}, "
                    f"using longest coding transcript {transcript.id}"
                )
            else:
                logger.debug(f"No coding transcripts for {gene_name}")
                skipped += 1
                continue

        transcripts.append(transcript)

    logger.info(
        f"Resolved {len(transcripts)} MANE transcripts "
        f"(skipped {skipped}) via NCBI fallback"
    )
    return transcripts


def _process_transcript(transcript, model, device, gencode_version, batch_size):
    """Predict NMD efficiency at every codon in *transcript*.

    Returns a list of (chrom, genomic_pos, prediction) tuples — one per CDS
    nucleotide — with each codon's three nucleotides sharing the mean
    prediction across the three stop-codon types.
    """
    cds_length = sum(len(c) for c in transcript.cdss)
    if cds_length > 30_000:
        logger.warning(
            f"Skipping {transcript.id}: CDS too long ({cds_length} nt)"
        )
        return []

    effective_batch_size = 1 if cds_length > 10_000 else batch_size

    tid = transcript.id.split(".")[0]
    try:
        wt_sequence = create_six_track_encoding(tid, gencode_version=gencode_version)
    except Exception as e:
        logger.debug(f"Skipping {transcript.id}: encoding failed ({e})")
        sleep(5)
        return []

    utr5_length = sum(len(u) for u in transcript.utr5s)
    n_codons = cds_length // 3
    genomic_pos_map = _cds_to_genomic_positions(transcript)
    chrom = transcript.chrom

    # Predict for all 3 stop codons at every codon position
    stop_predictions: dict[int, list[float]] = {}
    for stop_codon, stop_ohe in _STOP_OHE.items():
        sequences = []
        valid_indices = []
        for ci in range(n_codons):
            ptc_pos = utr5_length + ci * 3
            if ptc_pos + 3 > len(wt_sequence):
                break
            mutated = wt_sequence.copy()
            mutated[ptc_pos : ptc_pos + 3, :4] = stop_ohe
            sequences.append(mutated)
            valid_indices.append(ci)

        if not sequences:
            continue

        preds = _predict_sequences_batched(
            model, sequences, device, effective_batch_size
        )
        for ci, p in zip(valid_indices, preds):
            stop_predictions.setdefault(ci, []).append(p)

    # Average across stop codons → one value per codon → expand to 3 nt
    results = []
    for ci in range(n_codons):
        preds = stop_predictions.get(ci)
        if preds is None:
            continue
        mean_pred = sum(preds) / len(preds)
        for nt_offset in range(3):
            cds_nt = ci * 3 + nt_offset
            if cds_nt < len(genomic_pos_map):
                gpos = genomic_pos_map[cds_nt]
                results.append((chrom, gpos, mean_pred))

    return results


def _get_reachable_stops(codon_ohe):
    """Return the set of stop codons reachable by a single-nucleotide change.

    Args:
        codon_ohe: (3, 4) array — one-hot encoded codon (first 4 columns).

    Returns:
        List of unique stop codon strings reachable by exactly one SNV
        (e.g. ``["TAG", "TGA"]``).  Empty if the codon is already a stop.
    """
    wt_bases = [_INDEX_NUC[int(np.argmax(codon_ohe[i]))] for i in range(3)]
    wt_codon = "".join(wt_bases)
    if wt_codon in _STOP_CODONS:
        return []

    reachable = set()
    for pos in range(3):
        for alt in "ACGT":
            if alt == wt_bases[pos]:
                continue
            mut_bases = list(wt_bases)
            mut_bases[pos] = alt
            mut_codon = "".join(mut_bases)
            if mut_codon in _STOP_CODONS:
                reachable.add(mut_codon)
    return sorted(reachable)


def _process_transcript_snv(transcript, model, device, gencode_version, batch_size):
    """Predict NMD for every possible stop-gain SNV in *transcript*.

    For each codon, determines which stop codons are reachable by a single
    nucleotide change and predicts once per reachable stop type.  Each
    (codon, stop_type) pair becomes a separate set of entries so the BED
    carries per-stop-type predictions.

    Returns a list of dicts with keys: chrom, pos, prediction, gene_name,
    transcript_id, aa_position, ref_codon, mut_codon, strand.
    """
    cds_length = sum(len(c) for c in transcript.cdss)
    if cds_length > 30_000:
        logger.warning(
            f"Skipping {transcript.id}: CDS too long ({cds_length} nt)"
        )
        return []

    effective_batch_size = 1 if cds_length > 10_000 else batch_size

    tid = transcript.id.split(".")[0]
    try:
        wt_sequence = create_six_track_encoding(tid, gencode_version=gencode_version)
    except Exception as e:
        logger.debug(f"Skipping {transcript.id}: encoding failed ({e})")
        sleep(5)
        return []

    utr5_length = sum(len(u) for u in transcript.utr5s)
    n_codons = cds_length // 3
    genomic_pos_map = _cds_to_genomic_positions(transcript)
    chrom = transcript.chrom
    strand = transcript.strand
    gene_name = transcript.gene.name or tid

    # Collect one sequence per (codon, reachable stop type)
    sequences = []
    seq_info: list[tuple[int, str, str]] = []  # (codon_idx, ref_codon, mut_codon)

    for ci in range(n_codons):
        ptc_pos = utr5_length + ci * 3
        if ptc_pos + 3 > len(wt_sequence):
            break

        codon_ohe = wt_sequence[ptc_pos : ptc_pos + 3, :4]
        reachable = _get_reachable_stops(codon_ohe)
        if not reachable:
            continue

        ref_codon = "".join(
            _INDEX_NUC[int(np.argmax(codon_ohe[j]))] for j in range(3)
        )

        for stop in reachable:
            mutated = wt_sequence.copy()
            mutated[ptc_pos : ptc_pos + 3, :4] = _STOP_OHE[stop]
            sequences.append(mutated)
            seq_info.append((ci, ref_codon, stop))

    if not sequences:
        return []

    # Predict all at once
    preds = _predict_sequences_batched(
        model, sequences, device, effective_batch_size
    )

    # One result per (codon, stop_type) with all genomic positions
    results = []
    for (ci, ref_codon, mut_codon), pred in zip(seq_info, preds):
        positions = []
        for nt_offset in range(3):
            cds_nt = ci * 3 + nt_offset
            if cds_nt < len(genomic_pos_map):
                positions.append(genomic_pos_map[cds_nt])
        if positions:
            results.append({
                "chrom": chrom,
                "positions": positions,
                "prediction": pred,
                "gene_name": gene_name,
                "transcript_id": transcript.id,
                "aa_position": ci + 1,
                "ref_codon": ref_codon,
                "mut_codon": mut_codon,
                "strand": strand,
            })

    return results


# ── Main entry point ─────────────────────────────────────────────────────────

@app.command()
def generate(
    gencode_version: str = GENCODE_VERSION,
    batch_size: int = 16,
    max_genes: int = 0,
    mode: str = typer.Option(
        "snv",
        help="'snv': predict each stop-gain SNV (faster, biologically realistic). "
             "'codon': replace entire codon with TAG/TAA/TGA and average.",
    ),
):
    """Generate genome-wide NMD predictions as BED + bigWig for MANE Select transcripts.

    Two prediction modes are available:

    - **snv** (default): For each codon, find every single-nucleotide change
      that creates a stop codon, mutate only that nucleotide, predict, and
      average across SNVs.  Codons with no possible stop-gain SNV are skipped,
      making this mode faster.

    - **codon**: Replace the entire codon with each of TAG / TAA / TGA, predict
      all three, and average.  Every codon gets a prediction.

    Args:
        gencode_version: GENCODE annotation version (default from config).
        batch_size: Inference batch size.
        max_genes: Process at most this many genes (0 = all; useful for testing).
        mode: Prediction mode — 'snv' or 'codon'.
    """
    if mode not in ("snv", "codon"):
        raise typer.BadParameter(f"mode must be 'snv' or 'codon', got '{mode}'")

    process_fn = _process_transcript_snv if mode == "snv" else _process_transcript
    logger.info(f"Prediction mode: {mode}")
    supp_dir = PROJ_ROOT / "manuscript" / "supplementary" / "files"
    supp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # ── 1. Model ──────────────────────────────────────────────────────
        config = TrainerConfig()
        model, device = _setup_model(config)
        load_model(model, MODELS_DIR / "NMDetectiveAI.pt", device=device)
        model.eval()
        logger.info(f"Model loaded on {device}")

        # ── 2. Genome ─────────────────────────────────────────────────────
        genome = Genome(gencode_version)
        logger.info(f"Found {sum(1 for g in genome.genes if g.type == 'protein_coding')} protein-coding genes")

        # ── 3. MANE Select transcripts ────────────────────────────────────
        mane_transcripts = _get_mane_select_transcripts(genome)

        if max_genes > 0:
            mane_transcripts = mane_transcripts[:max_genes]
            logger.info(f"Test mode: processing only {max_genes} genes")

        # ── 4. Predict every MANE transcript → raw bedGraph ───────────────
        raw_bg = tmp_dir / "raw.bedGraph"
        raw_bed = tmp_dir / "raw.bed" if mode == "snv" else None
        processed = 0
        skipped = 0
        total_entries = 0

        bed_fh = open(raw_bed, "w") if raw_bed else None
        try:
            with open(raw_bg, "w") as fh:
                for transcript in tqdm(mane_transcripts, desc="Processing MANE genes"):
                    gene_name = transcript.gene.name or transcript.id

                    try:
                        results = process_fn(
                            transcript, model, device, gencode_version, batch_size
                        )
                        if mode == "snv":
                            for r in results:
                                # bedGraph: one row per nucleotide
                                for gpos in r["positions"]:
                                    fh.write(
                                        f"{r['chrom']}\t{gpos}\t{gpos + 1}"
                                        f"\t{r['prediction']:.4f}\n"
                                    )
                                # BED: one row per (codon, stop type)
                                bed_start = min(r["positions"])
                                bed_end = max(r["positions"]) + 1
                                bed_fh.write(
                                    f"{r['chrom']}\t{bed_start}\t{bed_end}"
                                    f"\t{r['gene_name']}:{r['ref_codon']}{r['aa_position']}{r['mut_codon']}"
                                    f"\t{int(r['prediction'] * 1000)}"
                                    f"\t{r['strand']}"
                                    f"\t{bed_start}\t{bed_end}"
                                    f"\t0,0,0"
                                    f"\t{r['prediction']:.4f}"
                                    f"\t{r['gene_name']}"
                                    f"\t{r['transcript_id']}"
                                    f"\t{r['aa_position']}"
                                    f"\t{r['ref_codon']}"
                                    f"\t{r['mut_codon']}\n"
                                )
                        else:
                            for chrom, gpos, pred in results:
                                fh.write(f"{chrom}\t{gpos}\t{gpos + 1}\t{pred:.4f}\n")
                        total_entries += len(results)
                        if results:
                            processed += 1
                    except Exception as e:
                        logger.debug(f"Skipping {gene_name}: {e}")
                        skipped += 1
        finally:
            if bed_fh:
                bed_fh.close()

        logger.info(
            f"Processed {processed} genes, skipped {skipped}, "
            f"wrote {total_entries} bedGraph entries"
        )

        # ── 5. Sort and deduplicate (mean for overlapping genes) ──────────
        sorted_bg = tmp_dir / "sorted.bedGraph"
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", str(raw_bg), "-o", str(sorted_bg)],
            check=True,
        )

        deduped_bg = tmp_dir / "deduped.bedGraph"
        awk_script = (
            '{'
            '  key=$1"\\t"$2"\\t"$3;'
            '  if (key == prev) { sum += $4; n++ }'
            '  else {'
            '    if (NR > 1) print prev"\\t"sum/n;'
            '    prev = key; sum = $4; n = 1'
            '  }'
            '}'
            'END { if (n > 0) print prev"\\t"sum/n }'
        )
        with open(deduped_bg, "w") as out:
            subprocess.run(
                ["awk", "-F\t", awk_script, str(sorted_bg)],
                stdout=out,
                check=True,
            )
        logger.info("bedGraph sorted and deduplicated")

        # ── 6. Chromosome sizes ───────────────────────────────────────────
        chrom_sizes = tmp_dir / "hg38.chrom.sizes"
        urllib.request.urlretrieve(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
            str(chrom_sizes),
        )

        # ── 7. bedGraph → bigWig ─────────────────────────────────────────
        bw_path = supp_dir / "NMDetectiveAI_MANE.bw"
        subprocess.run(
            [_BEDGRAPH_TO_BIGWIG, str(deduped_bg), str(chrom_sizes), str(bw_path)],
            check=True,
        )
        logger.success(f"bigWig written to {bw_path}")

        # ── 8. Save BED with header ──────────────────────────────────────
        bed_path = supp_dir / "NMDetectiveAI_MANE.bed"
        if mode == "snv":
            # Sort enriched BED by chrom, start
            sorted_bed = tmp_dir / "sorted.bed"
            subprocess.run(
                ["sort", "-k1,1", "-k2,2n", str(raw_bed), "-o", str(sorted_bed)],
                check=True,
            )
            header = (
                "#chrom\tchromStart\tchromEnd\tname\tscore\tstrand"
                "\tthickStart\tthickEnd\titemRgb"
                "\tprediction\tgeneName\ttranscriptId\taaPosition\trefCodon\tmutCodon\n"
            )
            with open(bed_path, "w") as out_fh, open(sorted_bed) as in_fh:
                out_fh.write(header)
                shutil.copyfileobj(in_fh, out_fh)
        else:
            with open(bed_path, "w") as out_fh, open(deduped_bg) as in_fh:
                out_fh.write("#chrom\tchromStart\tchromEnd\tprediction\n")
                shutil.copyfileobj(in_fh, out_fh)
        logger.success(f"BED written to {bed_path}")

        # ── 9. Summary stats ─────────────────────────────────────────────
        n_lines = sum(1 for _ in open(deduped_bg))
        logger.info(f"Final track: {n_lines} unique genomic positions")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.command()
def generate_from_csvs(
    gencode_version: str = GENCODE_VERSION,
):
    """Generate bigWig from pre-computed per-gene CSV predictions in GW_2.

    Reads the per-transcript TAG-only PTC prediction CSVs produced by
    ``predict generate-all-mane-predictions``, maps CDS positions back to
    genomic coordinates, expands each codon to 3 nucleotides, and writes
    a bigWig + BED.
    """
    csv_dir = TABLES_DIR / "GW_2"
    supp_dir = PROJ_ROOT / "manuscript" / "supplementary" / "files"
    supp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # ── 1. Genome ─────────────────────────────────────────────────────
        genome = Genome(gencode_version)
        logger.info(f"Loaded genome {gencode_version}")

        # ── 2. Collect all CSV files ──────────────────────────────────────
        csv_files = sorted(csv_dir.glob("*_ptc_predictions.csv"))
        logger.info(f"Found {len(csv_files)} prediction CSVs in {csv_dir}")

        # ── 3. Process each CSV → bedGraph entries ────────────────────────
        raw_bg = tmp_dir / "raw.bedGraph"
        processed = 0
        skipped = 0
        total_entries = 0

        with open(raw_bg, "w") as fh:
            for csv_path in tqdm(csv_files, desc="Processing CSVs"):
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    logger.debug(f"Failed to read {csv_path.name}: {e}")
                    skipped += 1
                    continue

                if df.empty:
                    skipped += 1
                    continue

                tid_full = df["transcript_id"].iloc[0]
                tid_base = tid_full.split(".")[0]

                # Look up transcript
                try:
                    transcript = genome.transcripts[tid_base]
                except (KeyError, Exception):
                    logger.debug(f"Transcript {tid_full} not found in {gencode_version}")
                    skipped += 1
                    continue

                if not transcript.cdss:
                    logger.debug(f"Transcript {tid_full} has no CDS")
                    skipped += 1
                    continue

                utr5_length = sum(len(u) for u in transcript.utr5s) if transcript.utr5s else 0
                genomic_pos_map = _cds_to_genomic_positions(transcript)
                chrom = transcript.chrom

                for _, row in df.iterrows():
                    ptc_pos = int(row["ptc_position"])
                    pred = float(row["prediction"])

                    # Reverse: ptc_position = utr5_length + codon_idx*3 + 1
                    codon_idx = (ptc_pos - 1 - utr5_length) // 3
                    if codon_idx < 0:
                        continue

                    # Expand codon to 3 nucleotide positions
                    for nt_offset in range(3):
                        cds_nt = codon_idx * 3 + nt_offset
                        if cds_nt < len(genomic_pos_map):
                            gpos = genomic_pos_map[cds_nt]
                            fh.write(f"{chrom}\t{gpos}\t{gpos + 1}\t{pred:.4f}\n")
                            total_entries += 1

                processed += 1

        logger.info(
            f"Processed {processed} transcripts, skipped {skipped}, "
            f"wrote {total_entries} bedGraph entries"
        )

        # ── 4. Sort and deduplicate ───────────────────────────────────────
        sorted_bg = tmp_dir / "sorted.bedGraph"
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", str(raw_bg), "-o", str(sorted_bg)],
            check=True,
        )

        deduped_bg = tmp_dir / "deduped.bedGraph"
        awk_script = (
            '{'
            '  key=$1"\\t"$2"\\t"$3;'
            '  if (key == prev) { sum += $4; n++ }'
            '  else {'
            '    if (NR > 1) print prev"\\t"sum/n;'
            '    prev = key; sum = $4; n = 1'
            '  }'
            '}'
            'END { if (n > 0) print prev"\\t"sum/n }'
        )
        with open(deduped_bg, "w") as out:
            subprocess.run(
                ["awk", "-F\t", awk_script, str(sorted_bg)],
                stdout=out,
                check=True,
            )
        logger.info("bedGraph sorted and deduplicated")

        # ── 5. Chromosome sizes ───────────────────────────────────────────
        chrom_sizes = tmp_dir / "hg38.chrom.sizes"
        urllib.request.urlretrieve(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
            str(chrom_sizes),
        )

        # ── 6. bedGraph → bigWig ─────────────────────────────────────────
        bw_path = supp_dir / "NMDetectiveAI_GW.bw"
        subprocess.run(
            [_BEDGRAPH_TO_BIGWIG, str(deduped_bg), str(chrom_sizes), str(bw_path)],
            check=True,
        )
        logger.success(f"bigWig written to {bw_path}")

        # ── 7. Save BED with header ──────────────────────────────────────
        bed_path = supp_dir / "NMDetectiveAI_GW.bed"
        with open(bed_path, "w") as out_fh, open(deduped_bg) as in_fh:
            out_fh.write("#chrom\tchromStart\tchromEnd\tprediction\n")
            shutil.copyfileobj(in_fh, out_fh)
        logger.success(f"BED written to {bed_path}")

        # ── 8. Summary stats ─────────────────────────────────────────────
        n_lines = sum(1 for _ in open(deduped_bg))
        logger.info(f"Final track: {n_lines} unique genomic positions")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
