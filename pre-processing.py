from pdf2image import convert_from_path
from pydantic import BaseModel
from openai import Client
import polars as pl
import json
import ell
import os


client = Client(
    base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
)


class CharModel(BaseModel):
    name: str
    category: str
    year: int


class QModel(BaseModel):
    transcript: str
    contain_questions: bool


class QuestionFormat(BaseModel):
    question: str
    answers: list[str]
    correct_answer: str
    field: str


@ell.complex(
    model="openai/gpt-4o",
    client=client,
    response_format=QuestionFormat,
)
def formatting_question(question: str, date):
    return [
        ell.system(
            """Below is your task. For each given input containing:
            - A year
            - A multiple-choice question
            - Three answers (A, B, C) with exactly one correct answer enclosed in `<correct_answer>` tags

            You must output valid JSON with the following schema:
            - `"question": str` (The question. If it references a context that may have changed with time, insert the year into the question. Otherwise, keep it as is.)
            - `"answers": list[str]` (A list of possible answers, without any letters or numbering.)
            - `"correct_answer": str` (The correct answer, exactly matching the text in the `"answers"` list, without letters. Be sure to remove `<correct_answer>` tags and only keep the text.)
            - `"field": str` (One of the following exact values, chosen appropriately: `"Culture administrative et juridique"`, `"Finances publiques"`, `"Organisation, fonctionnement et politiques des institutions européennes"`, or `"Culture numérique"`)

            Important details:
            - Do not include "A.", "B.", or "C." in your output. Only the raw text of each answer.
            - The `correct_answer` must be the exact same text that appears in the `"answers"` list (no additional text or letters).
            - Insert the year into the question only if the question’s content is subject to change over time, like if there are proper nouns (e.g., small legislation that you know has changed since, job titles associated with a noun). If not, leave the question as is. Most of the time it is not necessary, but still, be cautious especially with proper nouns.
            - Make sure the output is strictly valid JSON (no trailing commas, no extra text).

            Use the following examples as guidelines:

            ---
            ### Example 1 (Basic case)

            **Input:**
            ```
            Date : 2020

            8. Le chef des armées est :
            A. le ministre des armées
            B. le Premier ministre
            <correct_answer>C. le Président de la République</correct_answer>
            ```

            **Output:**
            ```
            "question": "Le chef des armées est :"
            "answers": ["le ministre des armées","le Premier ministre","le Président de la République"]
            "correct_answer": "le Président de la République"
            "field": "Culture administrative et juridique"
            ```

            ---
            ### Example 2 (Basic case)

            **Input:**
            ```
            Date : 2020

            86. Le Conseil économique et social européen :
            A. est consulté pour l'élaboration de la législation
            <correct_answer>B.</correct_answer> veille au bon usage du budget
            C. est garant du droit
            ```

            **Output:**
            ```
            "question": "Le Conseil économique et social européen :"
            "answers": ["est consulté pour l'élaboration de la législation","veille au bon usage du budget","est garant du droit"]
            "correct_answer": "est consulté pour l'élaboration de la législation"
            "field": "Organisation, fonctionnement et politiques des institutions européennes"
            ```

            ---
            ### Example 3 (Outdated case)

            **Input:**
            ```
            Date : 2021

            103.Quelle autorité est chargée de lutter contre le piratage informatique ?
            A. ARCEP
            <correct_answer>B. HADOPI</correct_answer>
            C. Défenseur des droits
            ```

            **Output:**
            ```
            "question": "En 2021, quelle autorité était chargée de lutter contre le piratage informatique ?"
            "answers": ["ARCEP","HADOPI","Défenseur des droits"]
            "correct_answer": "HADOPI"
            "field": "Finances publiques"
            ```

            ---
            ### Example 4 (Outdated case)

            **Input:**
            ```
            Date : 2022

            49.Qui est secrétaire général du Gouvernement ?
            A. Marc Guillaume
            B. Stéphane Bouillon
            <correct_answer>C.</correct_answer> Claire Landais
            ```

            **Output:**
            ```
            "question": "En 2022, Qui est secrétaire général du Gouvernement ?"
            "answers": ["Marc Guillaume","Stéphane Bouillon","Claire Landais"]
            "correct_answer": "Claire Landais"
            "field": "Culture administrative et juridique"
            ```

            ---

            Now, given any similar input (year, question, choices with exactly one `<correct_answer>`), produce a single JSON object with the structure described above. Follow the examples precisely to handle time-sensitive questions by inserting the year when appropriate, and otherwise leaving it out. Ensure the `field` is selected from the four allowed categories."""
        ),
        ell.user(f"Date : {date}\n\n{question}"),
    ]


@ell.complex(
    model="google/gemini-2.0-flash-001",
    client=client,
    response_format=QModel,
)
def qcm(page):
    return [
        ell.system(
            """You are a AI-powered OCR system. Your result should respect the following format structure:
            transcript : A string containing the full exact transcript of the given page in markdown style, if any highlighting, circling, or coloring of the correct answer of a question is present, transcript it precisely using xml tag <correct_answer></correct_answer>, if not, transcript as is.
            contain_questions : A boolean indicating if the page contains multiple choice questions/answers to questions or not, NOT INCLUDING OTHER KIND OF EXERCISES OR EXPLANATIONS, identify them by seeing a numbered question and letter list of answers.
            For your information, the given page are from civil servant exams, the goal is to gather the page containing multiple choice questions for a dataset creation. Include only multiple choice questions and not other kind of exercises or tasks.
            Your output will go through pydantic, so ensure that the output is a valid dictionary."""
        ),
        ell.user(page),
    ]


"""
@ell.complex(model="openai/gpt-4o-mini", client=client, response_format=CharModel)
def get_charac(pages):
    return [
        ell.system(
            \"""Identify the characteristics (official name of competition, civil servant category, year) of the competition and extract them from the first pages of the document. Be accurate, and provide your answer using the given format structure :

                    "name": "Concours {métier} {interne/externe} {année}",
                    "category": "A", "B" or "C" (based on the hierarchical position)
                    "year": YYYY (use the latest year if 2 are mentionned)\"""
        ),
        ell.user(pages),
    ]





def get_source_dataframe(files_list):
    source_df = pl.DataFrame(
        schema={
            "id": pl.Int64,
            "year": pl.Int64,
            "competition": pl.Utf8,
            "source": pl.Utf8,
            "category": pl.Utf8,
            "file_name": pl.Utf8,
        }
    )
    print(source_df.head())
    return source_df

    for id in range(len(files_list)):
        error_id = []

        try:
            charac = dict(get_charac(convert_from_path(files_list[id])[2:]).parsed)
        except:
            error_id.append(id)
            continue
        print(charac)
        new_row = pl.DataFrame(
            {
                "id": pl.Series([id], dtype=pl.Int64),
                "year": pl.Series([charac["year"]], dtype=pl.Int64),
                "competition": pl.Series([charac["name"]], dtype=pl.Utf8),
                "source": pl.Series([files_list[id].split("/")[0]], dtype=pl.Utf8),
                "category": pl.Series([charac["category"]], dtype=pl.Utf8),
                "file_name": pl.Series([files_list[id]], dtype=pl.Utf8),
            }
        )
        source_df = pl.concat([source_df, new_row], how="vertical")
        print(source_df)

    source_df.write_csv("data.csv")()
"""


def get_full_files_list():
    os.chdir("data")
    folders_list = os.listdir()
    files_list = []
    for folder in folders_list:
        for file in os.listdir(folder):
            files_list.append(f"{folder}/{file}")
    return files_list


def transcript_files_list(files_list: list):
    for i in range(len(files_list)):
        transcript = ""
        print(f"file number = {i}/{len(files_list)}")
        print(f"file name = {files_list[i]}")
        for j in range(len(convert_from_path(files_list[i]))):
            print(f"page number = {j}/{len(convert_from_path(files_list[i]))}")
            current = convert_from_path(files_list[i])[j]
            transcript_payload = dict(qcm(current).parsed)
            print(transcript_payload["contain_questions"])
            if transcript_payload["contain_questions"] == True:
                transcript = transcript + "\n" + transcript_payload["transcript"]

        with open(f"{files_list[i].split('/')[1]}.txt", "w") as f:
            f.write(transcript)
    print("transcript files list completed")


def extract_questions_from_txt(files_list):
    full_data = []
    for file in files_list:
        with open(file, "r") as f:
            content = f.readlines()
        current_list = []
        current_name = file.split(".")[0].split("/")[1]
        i = 0
        temp = ""
        while i < len(content):
            if content[i] == "\n":
                i += 1
                current_list.append(temp)
                temp = ""
            else:
                temp += content[i]
                i += 1
        full_data.append({"name": current_name, "questions": current_list})
    with open("full_data.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f)
    return full_data


def create_question_table(questions):
    df = pl.DataFrame()
    id = 0
    for i, question_obj in enumerate(questions):
        for j, current_question in enumerate(question_obj["questions"]):
            question_name = question_obj["name"]
            date = question_name.split("-")[1]
            try:
                parsing = dict(formatting_question(current_question, date).parsed)
            except:
                parsing = dict(formatting_question(current_question, date).parsed)
            new_row = pl.DataFrame(
                {
                    "id": [id],
                    "question": [parsing["question"]],
                    "answers": [
                        parsing["answers"]
                    ],  # Wrap in [ ... ] to avoid expansion
                    "correct_answer": [parsing["correct_answer"]],
                    "field": [parsing["field"]],
                    "institution": [question_name.split("-")[0]],
                    "year": [int(question_name.split("-")[1])],
                    "session": [int(question_name.split("-")[2])],
                }
            )
            df = pl.concat([df, new_row], how="vertical")
            print(j)
            print(current_question)
            print(parsing["answers"])
            print(df.tail(4))
            id += 1
        print("Finished processing questions for", question_name)
        df.write_parquet(f"checkpoint_{i}.parquet")
    return df


def main():
    df = pl.read_parquet("ira_dataset.parquet")

    selected_df = df.select(["id", "answers", "correct_answer"])
    print(selected_df.head())

    for row in selected_df.iter_rows(named=True):
        if row["correct_answer"] not in row["answers"]:
            print(f"ID {row['id']}: correct_answer not found among answers")

    files_list = get_full_files_list()
    files_list = [item for item in files_list if "IRA" in item and ".txt" in item]
    print(files_list)
    questions = extract_questions_from_txt(files_list)
    i = 0
    for item in questions:
        for question in item["questions"]:
            i += 1
    print(i)


if __name__ == "__main__":
    main()
