import os
import re

OUTPUT_FILE_PATH_UNPROCESSED = "unprocessed_nouns.txt"
OUTPUT_FILE_PATH_PROCESSED = "nouns.txt"

FILES = [
    "A.txt", "B.txt", "C.txt", "Ç.txt", "D.txt", "DH.txt", "E.txt", "Ë.txt", "F.txt", "G.txt", "GJ.txt", "H.txt",
    "I.txt", "J.txt", "K.txt", "L.txt", "LL.txt", "M.txt", "N.txt", "NJ.txt", "O.txt", "P.txt", "Q.txt", "R.txt",
    "RR.txt", "S.txt", "SH.txt", "T.txt", "TH.txt", "U.txt", "V.txt", "X.txt", "XH.txt", "Y.txt", "Z.txt", "ZH.txt"
]

def process_nouns(input_file, processed_output_file, unprocessed_output_file):
    processed_nouns = []
    unprocessed_nouns = []

    input_file_path = os.path.join('fjalor', input_file)

    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        if ' m. ' in line or ' f. ' in line:
            if ' dhe ' in line or ' edhe ' in line:
                unprocessed_nouns.append(line)
            else:
                # Extract lemma and gender
                match = re.match(r'^([^,]+),~([mf]).*sh. ~([^,]+),', line)
                if match:
                    lemma = match.group(1).strip()
                    gender = 'm' if match.group(2) == 'm' else 'f'
                    n_pl = match.group(3).strip()

                    processed_nouns.append(f"lemma_n_{gender}={lemma}, n_{gender}_pl='{n_pl}'")

    # Write processed nouns to file
    processed_output_file_path = os.path.join('fjalor', processed_output_file)
    with open(processed_output_file_path, 'w', encoding='utf-8') as processed_file:
        for noun in processed_nouns:
            processed_file.write(noun + '\n')

    # Write unprocessed nouns to file
    unprocessed_output_file_path = os.path.join('fjalor', unprocessed_output_file)
    with open(unprocessed_output_file_path, 'w', encoding='utf-8') as unprocessed_file:
        for noun in unprocessed_nouns:
            unprocessed_file.write(noun + '\n')
    path = os.getcwd() + "/fjalor"
    for d_file in FILES:
        d_file_path = path + f"/{d_file}"
        process_nouns(d_file_path, processed_noun_file, unprocessed_nouns_file)

print("the list of extracted nouns is", processed_noun_file)
