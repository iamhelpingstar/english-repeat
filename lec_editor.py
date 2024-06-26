import re

cs = 285
lecture = "13"
part = "01"

filename = f"cs{cs}_{lecture}_{part}"


# 파일 열기
with open(f"{filename}.txt", "r", encoding="utf-8") as file:
    # 파일 내용 전체 읽기
    content = file.read()

content = content.replace("\n", "\n\n")

content = content.replace("\n\n\n", "\n\n")
content = content.replace("\n\n\n", "\n\n")


content = re.sub(
    r"\[p\.(\d{2})\]",
    lambda m: f"![{lecture}_{m.group(1)}](images/cs{cs}_{lecture}_{m.group(1)}.png)",
    content,
)

with open(f"{filename}.md", "w", encoding="utf-8") as file:
    file.write(content)
