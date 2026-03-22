table nmdetectiveAIPredictions
"NMDetective-AI stop-gain SNV NMD efficiency predictions"
    (
    string chrom;        "Chromosome"
    uint chromStart;     "Start position (0-based)"
    uint chromEnd;       "End position (exclusive)"
    string name;         "Gene|Ref>Alt|StopCodon"
    uint score;          "Prediction scaled to 0-1000"
    char[1] strand;      "Strand"
    uint thickStart;     "Thick draw start"
    uint thickEnd;       "Thick draw end"
    uint reserved;       "Item RGB color"
    float prediction;    "NMD efficiency prediction (0=triggered, 1=evading)"
    string geneName;     "Gene name"
    string transcriptId; "Transcript ID"
    uint aaPosition;     "Amino acid position (1-based)"
    string refCodon;     "Reference codon (mRNA)"
    )
