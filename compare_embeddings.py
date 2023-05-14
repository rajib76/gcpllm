import hashlib
import os
from typing import List

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from vertexai.language_models._language_models import TextEmbeddingModel

load_dotenv()
gcp_apl_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
gcp_project = os.getenv("GCP_PROJECT")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_apl_cred
os.environ["GCP_PROJECT"] = gcp_project
openai.api_key = os.getenv('api_key')
os.environ['TOKENIZERS_PARALLELISM'] = "false"


class EmbedTesting():
    def __init__(self):
        self.module = "Testing"

    def test_score(self, text1, text2, model="text-embedding-ada-002"):
        if model == "all-mpnet-base-v2":
            model = SentenceTransformer('all-mpnet-base-v2')
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
        elif model == "text-embedding-ada-002":
            emb = openai.Embedding.create(input=[text1, text2], engine=model, request_timeout=3)
            emb1 = np.asarray(emb.data[0]["embedding"])
            # print(emb1)
            emb2 = np.asarray(emb.data[1]["embedding"])
            # print(emb2)
        elif model == "textembedding-gecko@001":
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
            embeddigs = model.get_embeddings([text1])
            for embedding in embeddigs:
                emb1 = embedding.values
            embeddigs = model.get_embeddings([text2])
            for embedding in embeddigs:
                emb2 = embedding.values

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        score = cosine_similarity(emb1, emb2)

        return score


def main(embed_model:List,text_pairs:List):
    score_list = []
    model_list = []
    title_abstract_pairs = []

    et = EmbedTesting()
    title_abstracts=text_pairs
    for model in embed_model:
        for title_abstract_pair in title_abstracts:
            title = title_abstract_pair[0]
            abstract = title_abstract_pair[1]
            score = et.test_score(title,
                          abstract,
                          model)
            model_list.append(model)
            score_list.append(score)

            m = hashlib.md5()
            m.update(abstract.encode('UTF-8'))

            title_abstract_pairs.append(m.hexdigest())

    return model_list,score_list,title_abstract_pairs

if __name__=="__main__":
    embed_model = ['all-mpnet-base-v2','textembedding-gecko@001','text-embedding-ada-002']
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

    model_list,score_list,title_abstract_pairs = main(embed_model,title_abstracts)

    comparison_frame = pd.DataFrame({'Model': model_list,
                                     'Similarity_Score': score_list, 'Text_Pair': title_abstract_pairs})

    pd.set_option('display.max_columns', 3)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', 400)
    print(comparison_frame)



# et = EmbedTesting()
#
# title_abstracts = [[
#                        "Identification of Region-Specific Gene Isoforms in the Human Brain Using Long-Read Transcriptome Sequencing and Their Correlation with DNA Methylation",
#                        "Background: Site specificity is known in neuropsychiatric disorders, and differences in gene expression patterns could potentially explain this mechanism. However, studies using long-read transcriptome sequencing to analyze gene expression in different regions of the human brain have been limited, and none have focused on the hypothalamus, which plays a crucial role in regulating autonomic functions. Results: We performed long-read RNA sequencing on 12 samples derived from three different brain regions of the same individuals; the cerebellum, hypothalamus, and temporal cortex. We found that, compared to other regions, many genes with higher expression levels in the cerebellum and temporal cortex were associated with neuronal pathways, whereas those with higher expression levels in the hypothalamus were primarily linked to immune pathways. In addition, we investigated genes with different major isoforms in each brain region, even with similar overall expression levels among regions, and identified several genes, such as GAS7, that express different major isoforms in different regions. Many of these genes are involved in 'actin filament-based process' and 'cell projection organization' pathways, suggesting that region-dependent isoforms may have distinct roles in dendritic spine and neuronal formation in each region. Furthermore, we investigated the involvement of DNA methylation in these isoforms and found that DNA methylation may be associated with isoforms that have different first exons. Conclusions: Our results provide potentially valuable findings for future research on brain disorders and shed light on the mechanisms underlying isoform diversity in the human brain. Keywords: long-read transcriptome sequencing, hypothalamus, DNA methylation"
#                        ],
#                    ["Spatiotemporal cortical dynamics for rapid scene recognition as revealed by EEG decoding.",
#                     "The human visual system rapidly recognizes the categories and global properties of complex natural scenes. The present study investigated the spatiotemporal dynamics of neural signals involved in ultra-rapid scene recognition using electroencephalography (EEG) decoding. We recorded visual evoked potentials from 11 human observers for 232 natural scenes, each of which belonged to one of 13 natural scene categories (e.g., a bedroom or open country) and had three global properties (naturalness, openness, and roughness). We trained a deep convolutional classification model of the natural scene categories and global properties using EEGNet. Having confirmed that the model successfully classified natural scene categories and the three global properties, we applied Grad-CAM to the EEGNet model to visualize the EEG channels and time points that contributed to the classification. The analysis showed that EEG signals in the occipital lobes at short latencies (approximately 80~ ms) contributed to the classifications other than roughness, whereas those in the frontal lobes at relatively long latencies (~164 ms) contributed to the classification of naturalness and the individual scene category. These results suggest that different global properties are encoded in different cortical areas and with different timings, and that the encoding of scene categories shifts from the occipital to the frontal lobe over time."
#                     ],
#                    [
#                        "Self-assembly of CIP4 drives actin-mediated asymmetric pit-closing in clathrin-mediated endocytosis",
#                        "Clathrin-mediated endocytosis plays a pivotal role in signal transduction pathways between the extracellular environment and the intracellular space. Accumulating evidence from live-cell imaging and super-resolution microscopy of mammalian cells suggests an asymmetric distribution of actin fibers near the clathrin-coated pit, which induces asymmetric pit-closing, rather than radial constriction. However, detailed molecular mechanisms of this asymmetricity remain elusive. Herein, we used high-speed atomic force microscopy to demonstrate that CIP4, a multidomain protein with a classic F-BAR domain and intrinsically disordered regions, is necessary for asymmetric pit-closing. Strong self-assembly of CIP4 via intrinsically disordered regions, together with stereospecific interactions with the curved membrane and actin-regulating proteins, generates a small actin-rich environment near the pit, which deforms the membrane and closes the pit. Our results provide a mechanistic insight into how spatio-temporal actin polymerization near the plasma membrane is promoted by a collaboration of disordered and structured domains."]
#                    ]
#
# title_abstract_pairs = []
# score_list = []
# model = []
# for title_abstract_pair in title_abstracts:
#     title = title_abstract_pair[0]
#     abstract = title_abstract_pair[1]
#
#     score = et.test_score(title,
#                           abstract,
#                           "all-mpnet-base-v2")
#     model.append("all-mpnet-base-v2")
#     score_list.append(score)
#     title_abstract_pairs.append(title_abstract_pair)
#
#     score = et.test_score(title,
#                           abstract,
#                           "textembedding-gecko@001")
#     model.append("textembedding-gecko@001")
#     score_list.append(score)
#     title_abstract_pairs.append(title_abstract_pair)
#
#     score = et.test_score(title,
#                           abstract,
#                           "text-embedding-ada-002")
#     model.append("text-embedding-ada-002")
#     score_list.append(score)
#     title_abstract_pairs.append(title_abstract_pair)
#
# comparison_frame = pd.DataFrame({'Model': model,
#                                  'Similarity_Score': score_list, 'Text_Pair': title_abstract_pairs})
#
# pd.set_option('display.max_columns', 3)
# pd.set_option('max_colwidth', 100)
# pd.set_option('display.width', 400)
# print(comparison_frame)
