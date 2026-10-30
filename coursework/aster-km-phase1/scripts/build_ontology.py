"""Generate a small OWL/RDF organizational model from the corpus index."""

from __future__ import annotations

import csv
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://example.org/aster-km#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
for prefix, uri in [("rdf", RDF), ("rdfs", RDFS), ("owl", OWL), ("xsd", XSD), ("aster", BASE)]:
    ET.register_namespace(prefix, uri)

CLASSES = [
    "Document", "IncidentReport", "TechnicalGuide", "TestProcedure", "ReleaseNote",
    "Incident", "SoftwareSystem", "TestCase", "SoftwareRelease", "KnowledgeArea",
    "Concept", "SupportSpecialist", "SoftwareEngineer",
]
SUBCLASSES = {
    "IncidentReport": "Document",
    "TechnicalGuide": "Document",
    "TestProcedure": "Document",
    "ReleaseNote": "Document",
}
RELATIONS = {
    "documentsIncident": ("IncidentReport", "Incident"),
    "appliesToSystem": ("TechnicalGuide", "SoftwareSystem"),
    "specifiesTest": ("TestProcedure", "TestCase"),
    "describesRelease": ("ReleaseNote", "SoftwareRelease"),
    "aboutConcept": ("Document", "Concept"),
    "belongsToArea": ("Document", "KnowledgeArea"),
    "affectsSystem": ("Incident", "SoftwareSystem"),
    "reproducesIncident": ("TestCase", "Incident"),
    "resolvesIncident": ("SoftwareRelease", "Incident"),
    "assignedTo": ("Incident", "SoftwareEngineer"),
    "recordedBy": ("Incident", "SupportSpecialist"),
}
DATA_PROPERTIES = {
    "documentId": ("Document", "string"),
    "documentTitle": ("Document", "string"),
    "sourcePath": ("Document", "string"),
    "priorityLevel": ("Incident", "string"),
    "releaseVersion": ("SoftwareRelease", "string"),
}


def slug(value: str) -> str:
    return "".join(word.capitalize() for word in value.replace("-", " ").split())


def main() -> None:
    root = ET.Element(f"{{{RDF}}}RDF")
    ontology = ET.SubElement(root, f"{{{OWL}}}Ontology", {f"{{{RDF}}}about": BASE[:-1]})
    ET.SubElement(ontology, f"{{{RDFS}}}label").text = "Aster Systems organizational knowledge model"
    ET.SubElement(ontology, f"{{{RDFS}}}comment").text = (
        "Fictional academic data. Document categories and links correspond to the supplied corpus."
    )
    for name in CLASSES:
        element = ET.SubElement(root, f"{{{OWL}}}Class", {f"{{{RDF}}}about": BASE + name})
        ET.SubElement(element, f"{{{RDFS}}}label").text = name
        if name in SUBCLASSES:
            ET.SubElement(element, f"{{{RDFS}}}subClassOf", {
                f"{{{RDF}}}resource": BASE + SUBCLASSES[name]
            })
    for name, (domain, range_) in RELATIONS.items():
        element = ET.SubElement(root, f"{{{OWL}}}ObjectProperty", {f"{{{RDF}}}about": BASE + name})
        ET.SubElement(element, f"{{{RDFS}}}domain", {f"{{{RDF}}}resource": BASE + domain})
        ET.SubElement(element, f"{{{RDFS}}}range", {f"{{{RDF}}}resource": BASE + range_})
    for name, (domain, datatype) in DATA_PROPERTIES.items():
        element = ET.SubElement(root, f"{{{OWL}}}DatatypeProperty", {f"{{{RDF}}}about": BASE + name})
        ET.SubElement(element, f"{{{RDFS}}}domain", {f"{{{RDF}}}resource": BASE + domain})
        ET.SubElement(element, f"{{{RDFS}}}range", {f"{{{RDF}}}resource": XSD + datatype})

    kinds = {
        "Incident Report": ("IncidentReport", "documentsIncident", "Incident_"),
        "Technical Guide": ("TechnicalGuide", "appliesToSystem", "System_CommandSuite"),
        "Test Procedure": ("TestProcedure", "specifiesTest", "Test_"),
        "Release Note": ("ReleaseNote", "describesRelease", "Release_"),
    }

    def individual(identifier: str, class_name: str, links: dict[str, str] | None = None,
                   literals: dict[str, str] | None = None) -> None:
        node = ET.SubElement(root, f"{{{OWL}}}NamedIndividual", {f"{{{RDF}}}about": BASE + identifier})
        ET.SubElement(node, f"{{{RDF}}}type", {f"{{{RDF}}}resource": BASE + class_name})
        for property_name, target in (links or {}).items():
            ET.SubElement(node, f"{{{BASE}}}{property_name}", {f"{{{RDF}}}resource": BASE + target})
        for property_name, value in (literals or {}).items():
            element = ET.SubElement(node, f"{{{BASE}}}{property_name}", {
                f"{{{RDF}}}datatype": XSD + DATA_PROPERTIES[property_name][1]
            })
            element.text = value

    individual("System_CommandSuite", "SoftwareSystem")
    individual("Support_01", "SupportSpecialist")
    individual("Engineer_01", "SoftwareEngineer")
    for area in ["Support", "Operations", "Quality Assurance", "Engineering"]:
        individual("Area_" + slug(area), "KnowledgeArea")

    with (ROOT / "data" / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    concepts = sorted({row["concept"] for row in rows})
    release_versions = {
        row["concept"]: "2.4." + str(int(row["id"].split("-")[1]))
        for row in rows if row["category"] == "Release Note"
    }
    for concept in concepts:
        key = slug(concept)
        individual("Concept_" + key, "Concept")
        individual("Incident_" + key, "Incident", {
            "affectsSystem": "System_CommandSuite",
            "assignedTo": "Engineer_01",
            "recordedBy": "Support_01",
        })
        individual("Test_" + key, "TestCase", {"reproducesIncident": "Incident_" + key})
        individual("Release_" + key, "SoftwareRelease", {
            "resolvesIncident": "Incident_" + key
        }, {"releaseVersion": release_versions[concept]})
    for row in rows:
        class_name, predicate, target_prefix = kinds[row["category"]]
        target = target_prefix if class_name == "TechnicalGuide" else target_prefix + slug(row["concept"])
        individual("Doc_" + row["id"].replace("-", "_"), class_name, {
            "aboutConcept": "Concept_" + slug(row["concept"]),
            "belongsToArea": "Area_" + slug(row["area"]),
            predicate: target,
        }, {
            "documentId": row["id"],
            "documentTitle": row["title"],
            "sourcePath": "data/documents/" + row["file"],
        })
    destination = ROOT / "model" / "aster_ontology.owl"
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    parsed = ET.parse(destination).getroot()
    assert len(parsed.findall(f"{{{OWL}}}Class")) == 13
    assert len(parsed.findall(f"{{{OWL}}}ObjectProperty")) == 11
    assert len(parsed.findall(f"{{{OWL}}}NamedIndividual")) == 71
    print(f"OWL model: 13 classes, 11 object properties, 5 data properties, 71 individuals")


if __name__ == "__main__":
    main()

