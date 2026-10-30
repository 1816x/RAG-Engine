"""Render the life cycle, conceptual graph, and captured CLI output as PNG images."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidence"
DOCS.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
INK = "#182d45"
ACCENT = "#256b9c"
PALE = "#e7f2f9"
BG = "#f7f9fc"


def box(ax, x, y, w, h, title, subtitle="", fill=PALE, font=13):
    patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
                           boxstyle="round,pad=0.03,rounding_size=0.14",
                           linewidth=1.4, edgecolor="#6891b1",
                           facecolor=fill, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y+.13 if subtitle else y, title, ha="center", va="center",
            color=INK, weight="bold", fontsize=font, zorder=4)
    if subtitle:
        ax.text(x, y-.23, subtitle, ha="center", va="center",
                color="#4d6076", fontsize=10.2, zorder=4)


def arrow(ax, start, end, label=None, label_xy=None, color=ACCENT, rad=0):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                            linewidth=1.65, color=color,
                            connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(patch)
    if label:
        x, y = label_xy or ((start[0]+end[0])/2, (start[1]+end[1])/2)
        ax.text(x, y, label, color="#315674", fontsize=9.7, ha="center",
                va="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2), zorder=5)


def setup(figsize, limits):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, limits[0]); ax.set_ylim(0, limits[1])
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def lifecycle():
    fig, ax = setup((15, 8.2), (15, 8.2))
    ax.text(.55, 7.73, "Knowledge lifecycle at Aster Systems", color=INK,
            fontsize=22, weight="bold")
    ax.text(.55, 7.32, "Support experience becomes documented, searchable knowledge and returns as improved practice.",
            color="#52677d", fontsize=11.2)
    nodes = [
        (2.75,5.50,"1  Create or acquire","Support · Engineering"),
        (7.50,5.50,"2  Document and store","Ticket · guide · test · note"),
        (12.25,5.50,"3  Classify and link","Category · concept · area"),
        (12.25,2.85,"4  Share and retrieve","Search · review · handoff"),
        (7.50,2.85,"5  Apply knowledge","Operations · QA · release"),
        (2.75,2.85,"6  Update and improve","Feedback · version control"),
    ]
    for x,y,title,sub in nodes:
        box(ax,x,y,3.72,1.18,title,sub,font=12.3)
    arrow(ax,(4.65,5.5),(5.58,5.5))
    arrow(ax,(9.39,5.5),(10.33,5.5),"TF-IDF model",(9.86,5.92))
    arrow(ax,(12.25,4.88),(12.25,3.49),"cosine rank",(13.22,4.2))
    arrow(ax,(10.35,2.85),(9.39,2.85))
    arrow(ax,(5.61,2.85),(4.65,2.85))
    ax.plot([.60,.60,2.75],[2.85,6.55,6.55],color=ACCENT,lw=1.65,zorder=1)
    arrow(ax,(2.75,6.55),(2.75,6.11),"feedback",(1.15,6.76))
    ax.text(7.5,.82,"Tacit knowledge: troubleshooting judgment and test insight    →    Explicit knowledge: reports, guides, procedures, release notes",
            fontsize=10.3,color="#3f5e79",ha="center")
    fig.savefig(DOCS/"knowledge_lifecycle.png",dpi=200,facecolor=BG)
    plt.close(fig)


def graph():
    fig, ax = setup((16, 10.4), (16, 10.4))
    ax.text(.55,9.85,"Organizational knowledge graph",fontsize=22,weight="bold",color=INK)
    ax.text(.55,9.46,"Class relationships in the OWL model · arrows name object properties · document types inherit from Document",
            fontsize=10.7,color="#52677d")
    ax.text(.65,8.84,"DOCUMENT STRUCTURE",fontsize=11.2,weight="bold",color=ACCENT)
    box(ax,1.7,6.70,2.30,.86,"Document",font=12.6)
    rows=[
        (8.13,"Incident Report","documentsIncident","Incident"),
        (7.16,"Technical Guide","appliesToSystem","Software System"),
        (6.19,"Test Procedure","specifiesTest","Test Case"),
        (5.22,"Release Note","describesRelease","Software Release"),
    ]
    for y,doc,relation,target in rows:
        box(ax,5.55,y,2.54,.73,doc,font=11.5)
        box(ax,10.18,y,2.67,.73,target,fill="#f0f4ec",font=11.5)
        arrow(ax,(4.21,y),(2.92,6.70),"subclass of",(3.46,y+.35 if y<6.70 else y-.30),color="#81909f")
        arrow(ax,(6.85,y),(8.80,y),relation,(7.83,y+.28))
    box(ax,13.78,7.40,2.42,.73,"Concept",fill="#f0f4ec",font=11.4)
    box(ax,13.78,6.11,2.42,.73,"Knowledge Area",fill="#f0f4ec",font=11.4)
    ax.plot([1.7,1.7,13.78,13.78],[7.15,8.50,8.50,8.10],
            color="#7996ae",lw=1.5,zorder=1)
    arrow(ax,(13.78,8.10),(13.78,7.79),"aboutConcept",(11.76,8.51),color="#7996ae")
    ax.plot([1.7,1.7,13.78,13.78],[6.25,4.82,4.82,5.38],
            color="#7996ae",lw=1.5,zorder=1)
    arrow(ax,(13.78,5.38),(13.78,5.72),"belongsToArea",(11.70,4.82),color="#7996ae")
    ax.text(.65,4.45,"OPERATIONAL LINKS",fontsize=11.2,weight="bold",color=ACCENT)
    box(ax,8.0,2.65,2.28,.83,"Incident",font=12.3)
    box(ax,2.2,3.50,2.70,.74,"Test Case",font=11.3)
    box(ax,2.2,2.24,2.70,.74,"Software Release",font=11.3)
    box(ax,13.55,3.65,2.70,.74,"Software System",font=11.3)
    box(ax,13.55,2.65,2.70,.74,"Software Engineer",font=11.3)
    box(ax,13.55,1.65,2.70,.74,"Support Specialist",font=11.3)
    arrow(ax,(3.58,3.50),(6.81,2.88),"reproducesIncident",(5.25,3.49))
    arrow(ax,(3.58,2.24),(6.81,2.60),"resolvesIncident",(5.27,2.10))
    arrow(ax,(9.18,2.94),(12.13,3.57),"affectsSystem",(10.64,3.57))
    arrow(ax,(9.18,2.65),(12.13,2.65),"assignedTo",(10.63,2.89))
    arrow(ax,(9.18,2.37),(12.13,1.72),"recordedBy",(10.64,1.78))
    ax.text(.65,.60,"Example: IR-01 → Reconnect timeout → Incident_ReconnectTimeout → System_CommandSuite",
            fontsize=10.2,color="#52677d")
    fig.savefig(DOCS/"knowledge_graph.png",dpi=200,facecolor=BG)
    plt.close(fig)


def terminal_image(source_name: str, target_name: str, max_lines: int = 42):
    raw = (EVIDENCE/source_name).read_text(encoding="utf-8").splitlines()[:max_lines]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 19)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    width = 1720
    height = 88+len(raw)*32+26
    image = Image.new("RGB", (width,height), "#142237")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,width,62),fill="#20334c")
    draw.text((28,18),f"Captured execution output · {source_name}",font=title_font,fill="#e6f0f7")
    for n,line in enumerate(raw):
        draw.text((28,82+n*32),line[:130],font=font,fill="#d7e7f2")
    image.save(EVIDENCE/target_name)


if __name__ == "__main__":
    lifecycle()
    graph()
    terminal_image("demo.txt","demo_output.png")
    terminal_image("classification.txt","classification_output.png")
    terminal_image("search.txt","search_output.png")
    terminal_image("evaluate.txt","evaluation_output.png")
    terminal_image("tests.txt","test_output.png")
    print("Rendered lifecycle, graph, and three execution captures")

