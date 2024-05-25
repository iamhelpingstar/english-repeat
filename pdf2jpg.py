import fitz

cs = 285
lecture = "02"
filename = f"cs{cs}_{lecture}"
doc = fitz.open(f"pdf/{filename}.pdf")
for i, page in enumerate(doc):
    img = page.get_pixmap(dpi=200)
    img.save(f"images/{filename}_{i+1:02d}.png")
