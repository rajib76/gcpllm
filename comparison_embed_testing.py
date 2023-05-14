import csv
import hashlib
import importlib

scores = []
models = []
text_pairs = []
embedmodels = [{"class": "sent_embed", "modelId": "all-mpnet-base-v2"},
               {"class": "openai_embed", "modelId": "text-embedding-ada-002"},
               {"class": "palmai_embed", "modelId": "textembedding-gecko@001"},
               {"class": "cohere_embed", "modelId": "embed-multilingual-v2.0"},
               {"class": "cohere_embed", "modelId": "embed-english-v2.0"},
               {"class": "cohere_embed", "modelId": "embed-english-light-v2.0"}]
context_question_pairs = [
    ["Southern California, often abbreviated SoCal, is a geographic and cultural region that generally comprises "
     "California's southernmost 10 counties. The region is traditionally described as 'eight counties', "
     "based on demographics and economic ties: Imperial, Los Angeles, Orange, Riverside, San Bernardino, "
     "San Diego, Santa Barbara, and Ventura. The more extensive 10-county definition, including Kern and San Luis "
     "Obispo counties, is also used based on historical political divisions. Southern California is a major "
     "economic center for the state of California and the United States.", "Despite being traditionall described "
                                                                           "as 'eight counties', how many counties "
                                                                           "does this region actually have?"],
    ["The 8- and 10-county definitions are not used for the greater Southern California Megaregion, one of the 11 "
     "megaregions of the United States. The megaregion's area is more expansive, extending east into Las Vegas, "
     "Nevada, and south across the Mexican border into Tijuana.", "What is the name of the state that the "
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
     "population.", "Which of the three heavily populated areas has the least number of inhabitants?"]
]

context_question_pairs_shuffled = [
    ["Southern California, often abbreviated SoCal, is a geographic and cultural region that generally comprises "
     "California's southernmost 10 counties. The region is traditionally described as 'eight counties', "
     "based on demographics and economic ties: Imperial, Los Angeles, Orange, Riverside, San Bernardino, "
     "San Diego, Santa Barbara, and Ventura. The more extensive 10-county definition, including Kern and San Luis "
     "Obispo counties, is also used based on historical political divisions. Southern California is a major "
     "economic center for the state of California and the United States.", "What is the name of the state that the "
                                                                           "megaregion expands to in the east?"],
    ["The 8- and 10-county definitions are not used for the greater Southern California Megaregion, one of the 11 "
     "megaregions of the United States. The megaregion's area is more expansive, extending east into Las Vegas, "
     "Nevada, and south across the Mexican border into Tijuana.", "Which of the three heavily populated areas has the "
                                                                  "least number of inhabitants?"],
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
     "population.", "Despite being traditional described as 'eight counties', how many counties does this region "
                    "actually have?"]
]

if __name__ == "__main__":

    for embded in embedmodels:
        embed_model = embded["class"]
        model_id = embded["modelId"]
        embed_model = importlib.import_module("embedding"+".%s" %embed_model)
        embed_model_class = embed_model.Embedding(model_id)
        for contex_question in context_question_pairs_shuffled:
            models.append(model_id)
            context = contex_question[0]
            question = contex_question[1]
            m = hashlib.md5()
            m.update(question.encode('UTF-8'))
            text_pairs.append(m.hexdigest())
            score = embed_model_class.get_score(context,question)
            scores.append(score)

        rows = zip(models,scores,text_pairs)

        file_loc = "context_correct_questions_shuff.csv"
        with open(file_loc,"w",newline="",encoding="utf-8") as compfile:
            writer=csv.writer(compfile)
            for row in rows:
                writer.writerow(row)