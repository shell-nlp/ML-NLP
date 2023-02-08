import openai

# set api key
openai.api_key = "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"  # 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
# use GPT-3 to generate text
prompt = ""
while True:
    prompt = input("问题： ")
    if prompt == "q":
        break
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # print the generated text
    message = completions.choices[0].text.strip()
    print("回答：\n", message)
print("程序退出！")
