import json
import random

# Existing 15 entries (shortened for space — replace with your full set)
original_entries = [
    {
        "section": "IPC 279",
        "title": "Rash driving or riding on a public way",
        "description": "Whoever drives any vehicle on a public way so rashly or negligently as to endanger human life or to cause injury shall be punished with imprisonment up to 6 months or fine up to ₹1000 or both.",
        "max_punishment": "6 months imprisonment or ₹1000 fine",
        "category": "Traffic Offence"
    },
    {
        "section": "IPC 304A",
        "title": "Causing death by negligence",
        "description": "When death is caused by a rash or negligent act not amounting to culpable homicide, punishment is imprisonment up to 2 years or fine or both.",
        "max_punishment": "2 years imprisonment or fine",
        "category": "Negligence"
    },
    # Add all other 13 original entries here...
]

# Categories pool
categories = ["Cyber Crime", "Fraud", "Property Offence", "Personal Injury", "Traffic Offence", "Negligence", "Obstruction of Justice", "Procedure", "General"]

# Generate 50 additional dummy entries
additional_entries = []
for i in range(1, 51):
    template = random.choice(original_entries)
    new_entry = {
        "section": f"{template['section'].split()[0]} {random.randint(300, 900)}A{i}",
        "title": f"{template['title']} (Variant {i})",
        "description": f"{template['description']} [This is a variant case {i} used for demo purposes.]",
        "max_punishment": template["max_punishment"],
        "category": random.choice(categories)
    }
    additional_entries.append(new_entry)

# Combine and save as JSON
full_dataset = original_entries + additional_entries

with open("legal_knowledge_base_full.json", "w", encoding="utf-8") as f:
    json.dump(full_dataset, f, indent=2, ensure_ascii=False)

print("✅ 50 dummy entries added. Total entries:", len(full_dataset))
