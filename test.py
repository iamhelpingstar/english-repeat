import clipboard
import re


def is_time_format(text):
    # 정규 표현식을 사용하여 d:dd 또는 dd:dd 형식을 감지
    return bool(re.match(r"^\d{1,2}:\d{2}$", text))


while True:
    text = []
    while True:
        line = input()
        if line == "v":
            break
        if is_time_format(line):
            continue
        text.append(line)

    text = " ".join(text)
    clipboard.copy(text)
    print("COPIED")
