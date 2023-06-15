import os
import time
import openai
from collections import defaultdict

openai.api_key = os.getenv("OPENAI_API_KEY")

system_message = """You are a component to evaluate a specific question.
Evaluations are made in a fixed output format.

"""

instructions = {
    "numerical_value_0_1": """Rate the given question ""QUESTION"" with a value between 0 and 1 according to the given criterias.
Explain and reflect upon your choice.

Answer in the format:
explanation: EXPLANATION
value: VALUE

Where EXPLANATION is a detailed explanation why VALUE was chosen.
""",
    "category_tf": """Assign the given question ""QUESTION"" a specific category according to the given criterias.
Explain and reflect upon your choice.

Categories:
- true
- false

Answer in the format:
explanation: EXPLANATION
category: CATEGORY

Where EXPLANATION is a detailed explanation why CATEGORY was chosen.
""",
}

didactic_criterias = {
    "simple_language": """The final decision-making basis for the formulation are
the subjects! A balance between unambiguity and simplicity
must be balanced!
- Formulate briefly, understandably and sufficiently precisely.
- Not bureaucratic, technocratic or scientific.
- Avoid foreign words
- Address the target group
- Use simple language, without slang, dialect
or subculture language
- Make the item precise in terms of the question's aim and intention.
""",
    "simple_and_positive_questions": """- Negations are linguistically negatively formulated questions.
Double negatives (especially in translations) are to be avoided in any case.
s) should be avoided at all costs.
- Avoid negative polarity: Item polarity (scale direction)
If possible, do not change. If negative and positive
and positive items are present, a separate evaluation is recommended.
separate evaluation.
- Unambiguity: Only assign a factual content or thought to each item!
or thought! One-dimensionality means,
that an agreement or disagreement allows only one interpretation.
interpretation. Ambiguity is to be avoided, especially with
especially in answer categories, not only in the formulation of questions.
question wording. it can also be said that no lo-
links should be present in a question (and, or...).
(and, or...).
""",
    "insinuation_free": """Insinuations and suggestive questions should be avoided.
Insinuations do not only refer to the proband, but can also address third parties.
A question may be rejected because the respondent does not agree with the insinuation.
Suggestive questions restrict the freedom of the answer and restrict the respondent's freedom of response.
Do not ask suggestive questions in a way that influences answering behavior.
""",
    "clear_temporal_reference": """Questions have a clear temporal reference.
Dates and timespans are precisely stated.
""",
    "concise_categories": """Use answer categories that are concise.
Closed questions should have disjunctive (non-overlapping) answer categories.
Categories that are precise and cover the solution space.
""",
    "explanation_of_the_unknown": """Clarify unclear terms and necessary knowledge for understanding the question.
""",
    "non_judgemental": """Avoid terms that are associated with strong opinions or emotions.
Strongly value-laden terms (peace, war, crime, justice) should be avoided.
They provoke more extreme response behavior.
""",
    "comparison_with_valid_scale": """Compared concepts are of the same scale.
You can't equate "higher" with "colder," or
"degree of rejection" with "brightness". "Do you prefer the
Grand Coalition or a Green policy?" cannot be compared unambiguously. The respondent must interpret for himself
how he or she matches a grand coalition and a potentially green policy. One is a party constellation for forming a government, the other a political program.
""",
}

shots = {
    "simple_language": {
        "category_tf": [
            {
                "user": """How can arrays be effectively utilized to store and retrieve multiple pieces of data in a computer program, ensuring simplicity and clarity while catering to the needs of the intended audience?
""",
                "assistant": """explanation: The question is long, uses the bureacratic word: "catering" and is unspecific about how to utilize an array.
It is also not clear if the question is about storing or retrieving data, and which abstraction level. Also what is meant with "pieces" of data.
category: false
""",
            }
        ]
    }
}

questions = [
    """Which of the following statements about binary trees is true?
- Binary trees can include Cliques
- A binary tree node can reference itself
- Binary trees are traversable""",
    """In the realm of computational structures, envision a parallelism between the intricate labyrinthine pathways of a dendrological marvel and a digital construct known as a binary tree. Assume the role of a knowledgeable purveyor in the domain of computer science, tasked with formulating an inquisitive conundrum, veiled in an abstruse tapestry of language. Considering the idiosyncrasies inherent in this task, might one endeavor to expound upon the enigmatic nuances of traversing the hallowed nodes and delicate edges of a binary tree, elucidating the elusive interplay between intricate branching and hierarchical representation?
""",
    """Hey y'all, listen up! I got a question about that fancy-schmancy Shor's algorithm for all you comp sci enthusiasts out there. Now, let's get down to business:

"Yo, what's the dealio with Shor's algorithm? How does it bust through them prime numbers like a boss and make quantum computing look like magic? Can you break it down for us regular folks without all them highfalutin terms?"
        """,
    """How can the magical world of binary trees unlock the secrets of the digital realm and empower our computational journeys?
""",
    "What is the definition of a biological device?",
]

didactic_criteria = "simple_language"
instruction = "category_tf"


def compose_message_one_shot(
    system_message,
    instructions,
    instruction,
    didactic_criterias,
    didactic_criteria,
    shots,
    shot_nr,
    question,
):
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": instructions[instruction]
            + "\nCriterias:\n"
            + didactic_criterias[didactic_criteria]
            + "\n---\nQuestion:\n"
            + shots[didactic_criteria][instruction][shot_nr]["user"],
        },
        {
            "role": "assistant",
            "content": shots[didactic_criteria][instruction][shot_nr]["assistant"],
        },
        {
            "role": "user",
            "content": instructions[instruction]
            + "\nCriterias:\n"
            + didactic_criterias[didactic_criteria]
            + '\n---\nQuestion:\n""\n'
            + question
            + '\n""',
        },
    ]


def compose_message_zero_shot(
    system_message,
    instructions,
    instruction,
    didactic_criterias,
    didactic_criteria,
    question,
):
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": instructions[instruction]
            + "\nCriterias:\n"
            + didactic_criterias[didactic_criteria]
            + '\n---\nQuestion:\n""\n'
            + question
            + '\n""',
        },
    ]


def compose_message_shot(
    instructions, instruction, didactic_criterias, didactic_criteria, shots, shot_nr
):
    return [
        {
            "role": "user",
            "content": instructions[instruction]
            + "\nCriterias:\n"
            + didactic_criterias[didactic_criteria]
            + "\n---\nQuestion:\n"
            + shots[didactic_criteria][instruction][shot_nr]["user"],
        },
        {
            "role": "assistant",
            "content": shots[didactic_criteria][instruction][shot_nr]["assistant"],
        },
    ]


def evaluate_all_criteria_one_shot(question):
    completions = []
    for didactic_criteria in didactic_criterias:
        zero_shot_message = compose_message_zero_shot(
            system_message,
            instructions,
            instruction,
            didactic_criterias,
            didactic_criteria,
            question,
        )
        shot_message = compose_message_shot(
            instructions, instruction, didactic_criterias, "simple_language", shots, 0
        )
        composed_message = [
            zero_shot_message[0],
            shot_message[0],
            shot_message[1],
            zero_shot_message[1],
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=composed_message
        )
        message_content = completion["choices"][0]["message"]["content"]
        completions.append(didactic_criteria + "\n---" + message_content)
        time.sleep(17)
    return completions

def evaluate_all_criteria_zero_shot(question):
    completions = []
    for didactic_criteria in didactic_criterias:
        zero_shot_message = compose_message_zero_shot(
            system_message,
            instructions,
            instruction,
            didactic_criterias,
            didactic_criteria,
            question,
        )
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=zero_shot_message
        )
        message_content = completion["choices"][0]["message"]["content"]
        completions.append(didactic_criteria + "\n---" + message_content)
        time.sleep(17)
    return completions


print("One-shot test:")
print(questions[0] + "\n---")
print(evaluate_all_criteria_one_shot(questions[0]))


print("Zero-shot test:")
print(questions[0] + "\n---")
print(evaluate_all_criteria_zero_shot(questions[0]))