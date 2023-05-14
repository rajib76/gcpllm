import csv
import hashlib
import importlib

if __name__=="__main__":
    # embedmodels = [{"class":"cohere_embed","modelId":"small"}]
    # embedmodels = [{"class":"openai_embed", "modelId":"text-embedding-ada-002"}]
    #embedmodels = [{"class": "palmai_embed", "modelId": "textembedding-gecko@001"}]
    scores=[]
    models=[]
    text_pairs=[]
    embedmodels = [{"class": "sent_embed", "modelId": "all-mpnet-base-v2"},
                   {"class":"openai_embed", "modelId":"text-embedding-ada-002"},
                   {"class": "palmai_embed", "modelId": "textembedding-gecko@001"},
                   {"class":"cohere_embed","modelId":"embed-multilingual-v2.0"},
                   {"class":"cohere_embed","modelId":"embed-english-v2.0"},
                   {"class":"cohere_embed","modelId":"embed-english-light-v2.0"}]
    title_abstracts = [[
        "Identification of Region-Specific Gene Isoforms in the Human Brain Using Long-Read Transcriptome Sequencing and Their Correlation with DNA Methylation",
        "Background: Site specificity is known in neuropsychiatric disorders, and differences in gene expression patterns could potentially explain this mechanism. However, studies using long-read transcriptome sequencing to analyze gene expression in different regions of the human brain have been limited, and none have focused on the hypothalamus, which plays a crucial role in regulating autonomic functions. Results: We performed long-read RNA sequencing on 12 samples derived from three different brain regions of the same individuals; the cerebellum, hypothalamus, and temporal cortex. We found that, compared to other regions, many genes with higher expression levels in the cerebellum and temporal cortex were associated with neuronal pathways, whereas those with higher expression levels in the hypothalamus were primarily linked to immune pathways. In addition, we investigated genes with different major isoforms in each brain region, even with similar overall expression levels among regions, and identified several genes, such as GAS7, that express different major isoforms in different regions. Many of these genes are involved in 'actin filament-based process' and 'cell projection organization' pathways, suggesting that region-dependent isoforms may have distinct roles in dendritic spine and neuronal formation in each region. Furthermore, we investigated the involvement of DNA methylation in these isoforms and found that DNA methylation may be associated with isoforms that have different first exons. Conclusions: Our results provide potentially valuable findings for future research on brain disorders and shed light on the mechanisms underlying isoform diversity in the human brain. Keywords: long-read transcriptome sequencing, hypothalamus, DNA methylation"
    ],
        ["Spatiotemporal cortical dynamics for rapid scene recognition as revealed by EEG decoding.",
         "The human visual system rapidly recognizes the categories and global properties of complex natural scenes. The present study investigated the spatiotemporal dynamics of neural signals involved in ultra-rapid scene recognition using electroencephalography (EEG) decoding. We recorded visual evoked potentials from 11 human observers for 232 natural scenes, each of which belonged to one of 13 natural scene categories (e.g., a bedroom or open country) and had three global properties (naturalness, openness, and roughness). We trained a deep convolutional classification model of the natural scene categories and global properties using EEGNet. Having confirmed that the model successfully classified natural scene categories and the three global properties, we applied Grad-CAM to the EEGNet model to visualize the EEG channels and time points that contributed to the classification. The analysis showed that EEG signals in the occipital lobes at short latencies (approximately 80~ ms) contributed to the classifications other than roughness, whereas those in the frontal lobes at relatively long latencies (~164 ms) contributed to the classification of naturalness and the individual scene category. These results suggest that different global properties are encoded in different cortical areas and with different timings, and that the encoding of scene categories shifts from the occipital to the frontal lobe over time."
         ],
        [
            "Self-assembly of CIP4 drives actin-mediated asymmetric pit-closing in clathrin-mediated endocytosis",
            "Clathrin-mediated endocytosis plays a pivotal role in signal transduction pathways between the extracellular environment and the intracellular space. Accumulating evidence from live-cell imaging and super-resolution microscopy of mammalian cells suggests an asymmetric distribution of actin fibers near the clathrin-coated pit, which induces asymmetric pit-closing, rather than radial constriction. However, detailed molecular mechanisms of this asymmetricity remain elusive. Herein, we used high-speed atomic force microscopy to demonstrate that CIP4, a multidomain protein with a classic F-BAR domain and intrinsically disordered regions, is necessary for asymmetric pit-closing. Strong self-assembly of CIP4 via intrinsically disordered regions, together with stereospecific interactions with the curved membrane and actin-regulating proteins, generates a small actin-rich environment near the pit, which deforms the membrane and closes the pit. Our results provide a mechanistic insight into how spatio-temporal actin polymerization near the plasma membrane is promoted by a collaboration of disordered and structured domains."]
    ]

    title_abstracts_wrong = [[
        "Spatiotemporal cortical dynamics for rapid scene recognition as revealed by EEG decoding.",
        "Background: Site specificity is known in neuropsychiatric disorders, and differences in gene expression patterns could potentially explain this mechanism. However, studies using long-read transcriptome sequencing to analyze gene expression in different regions of the human brain have been limited, and none have focused on the hypothalamus, which plays a crucial role in regulating autonomic functions. Results: We performed long-read RNA sequencing on 12 samples derived from three different brain regions of the same individuals; the cerebellum, hypothalamus, and temporal cortex. We found that, compared to other regions, many genes with higher expression levels in the cerebellum and temporal cortex were associated with neuronal pathways, whereas those with higher expression levels in the hypothalamus were primarily linked to immune pathways. In addition, we investigated genes with different major isoforms in each brain region, even with similar overall expression levels among regions, and identified several genes, such as GAS7, that express different major isoforms in different regions. Many of these genes are involved in 'actin filament-based process' and 'cell projection organization' pathways, suggesting that region-dependent isoforms may have distinct roles in dendritic spine and neuronal formation in each region. Furthermore, we investigated the involvement of DNA methylation in these isoforms and found that DNA methylation may be associated with isoforms that have different first exons. Conclusions: Our results provide potentially valuable findings for future research on brain disorders and shed light on the mechanisms underlying isoform diversity in the human brain. Keywords: long-read transcriptome sequencing, hypothalamus, DNA methylation"
    ],
        ["Self-assembly of CIP4 drives actin-mediated asymmetric pit-closing in clathrin-mediated endocytosis",
         "The human visual system rapidly recognizes the categories and global properties of complex natural scenes. The present study investigated the spatiotemporal dynamics of neural signals involved in ultra-rapid scene recognition using electroencephalography (EEG) decoding. We recorded visual evoked potentials from 11 human observers for 232 natural scenes, each of which belonged to one of 13 natural scene categories (e.g., a bedroom or open country) and had three global properties (naturalness, openness, and roughness). We trained a deep convolutional classification model of the natural scene categories and global properties using EEGNet. Having confirmed that the model successfully classified natural scene categories and the three global properties, we applied Grad-CAM to the EEGNet model to visualize the EEG channels and time points that contributed to the classification. The analysis showed that EEG signals in the occipital lobes at short latencies (approximately 80~ ms) contributed to the classifications other than roughness, whereas those in the frontal lobes at relatively long latencies (~164 ms) contributed to the classification of naturalness and the individual scene category. These results suggest that different global properties are encoded in different cortical areas and with different timings, and that the encoding of scene categories shifts from the occipital to the frontal lobe over time."
         ],
        [
            "Identification of Region-Specific Gene Isoforms in the Human Brain Using Long-Read Transcriptome Sequencing and Their Correlation with DNA Methylation",
            "Clathrin-mediated endocytosis plays a pivotal role in signal transduction pathways between the extracellular environment and the intracellular space. Accumulating evidence from live-cell imaging and super-resolution microscopy of mammalian cells suggests an asymmetric distribution of actin fibers near the clathrin-coated pit, which induces asymmetric pit-closing, rather than radial constriction. However, detailed molecular mechanisms of this asymmetricity remain elusive. Herein, we used high-speed atomic force microscopy to demonstrate that CIP4, a multidomain protein with a classic F-BAR domain and intrinsically disordered regions, is necessary for asymmetric pit-closing. Strong self-assembly of CIP4 via intrinsically disordered regions, together with stereospecific interactions with the curved membrane and actin-regulating proteins, generates a small actin-rich environment near the pit, which deforms the membrane and closes the pit. Our results provide a mechanistic insight into how spatio-temporal actin polymerization near the plasma membrane is promoted by a collaboration of disordered and structured domains."]
    ]


    q_a_pairs = [
        ["Southern California, often abbreviated SoCal, is a geographic and cultural region that generally comprises "
         "California's southernmost 10 counties. The region is traditionally described as 'eight counties', "
         "based on demographics and economic ties: Imperial, Los Angeles, Orange, Riverside, San Bernardino, "
         "San Diego, Santa Barbara, and Ventura. The more extensive 10-county definition, including Kern and San Luis "
         "Obispo counties, is also used based on historical political divisions. Southern California is a major "
         "economic center for the state of California and the United States.","Despite being traditionall described "
                                                                              "as 'eight counties', how many counties "
                                                                              "does this region actually have?"],
        ["The 8- and 10-county definitions are not used for the greater Southern California Megaregion, one of the 11 "
         "megaregions of the United States. The megaregion's area is more expansive, extending east into Las Vegas, "
         "Nevada, and south across the Mexican border into Tijuana.","What is the name of the state that the "
                                                                     "megaregion expands to in the east?"],
        ["Southern California includes the heavily built-up urban area stretching along the Pacific coast from "
         "Ventura, through the Greater Los Angeles Area and the Inland Empire, and down to Greater San Diego. "
         "Southern California's population encompasses seven metropolitan areas, or MSAs: the Los Angeles "
         "metropolitan area, consisting of Los Angeles and Orange counties; the Inland Empire, consisting of "
         "Riverside and San Bernardino counties; the San Diego metropolitan area; the Oxnard–Thousand Oaks–Ventura "
         "metropolitan area; the Santa Barbara metro area; the San Luis Obispo metropolitan area; and the El Centro "
         "area. Out of these, three are heavy populated areas: the Los Angeles area with over 12 million inhabitants, "
         "the Riverside-San Bernardino area with over four million inhabitants, and the San Diego area with over 3 "
         "million inhabitants. For CSA metropolitan purposes, the five counties of Los Angeles, Orange, Riverside, "
         "San Bernardino, and Ventura are all combined to make up the Greater Los Angeles Area with over 17.5 million "
         "people. With over 22 million people, southern California contains roughly 60 percent of California's "
         "population.","Which of the three heavily populated areas has the least number of inhabitants?"]
]

    for embded in embedmodels:
        embed_model = embded["class"]
        model_id = embded["modelId"]
        embed_model = importlib.import_module("embedding"+".%s" %embed_model)
        embed_model_class = embed_model.Embedding(model_id)
        for title_abstract_pair in title_abstracts_wrong:
            models.append(model_id)
            title = title_abstract_pair[0]
            abstract = title_abstract_pair[1]
            m = hashlib.md5()
            m.update(abstract.encode('UTF-8'))
            text_pairs.append(m.hexdigest())
            score = embed_model_class.get_score(title,abstract)
            scores.append(score)

        rows = zip(models,scores,text_pairs)

        file_loc = "comparisons_incorrect_abstracts.csv"
        with open(file_loc,"w",newline="",encoding="utf-8") as compfile:
            writer=csv.writer(compfile)
            for row in rows:
                writer.writerow(row)


