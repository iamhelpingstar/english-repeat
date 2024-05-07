import clipboard


while True:
    text = []

    line = input()
    while line != "v":
        text.append(line)
        line = input()
    text = " ".join(text)
    clipboard.copy(text)
    print("COPIED")
