import ollama


MODEL_NAME = "llama3.2:latest"


def ask_llm(prompt):

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    print("====================================")
    print("SafeVision AI LLM Test")
    print("====================================")

    prompt = """
You are a construction-site safety assistant.

Answer the following question using ONLY the
information provided below.

SAFETY POLICY:

A worker entering the controlled area should wear
an appropriate safety helmet.

If a worker is detected without the required helmet,
the worker should not be considered fully PPE compliant.

The site safety procedure should be followed when
a required helmet is missing.

QUESTION:

What should happen if a worker does not have
a safety helmet?

Do not invent additional safety rules.
"""

    print()
    print("QUESTION:")
    print(
        "What should happen if a worker does not "
        "have a safety helmet?"
    )

    print()
    print("LLM ANSWER:")
    print("------------------------------------")

    answer = ask_llm(prompt)

    print(answer)
