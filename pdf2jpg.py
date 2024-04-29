import fitz

lecture = "09"
filename = f"lec_{lecture}"
doc = fitz.open(f"{filename}.pdf")
for i, page in enumerate(doc):
    img = page.get_pixmap()
    img.save(f"images/{filename}_{i+1:02d}.png")
    print(
        f"![{filename}_{i+1:02d}](../../assets/images/rl/cs285/{filename}/{filename}_{i+1:02d}.png)"
    )
