import re

def extract_cite_content_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r'\\cite\{(.*?)\}'
    raw_matches = re.findall(pattern, content)
    matches = [ref.strip() for match in raw_matches for ref in match.split(',')]
    print(len(set(matches)))
    return set(matches)  

def extract_bib_entries(bib_filename, cite_keys):
    with open(bib_filename, 'r', encoding='utf-8') as file:
        content = file.readlines()

    inside_entry = False
    current_key = None
    saved_entries = []
    temp_entries = []   

    for line in content:
        if line.startswith('@'):
            inside_entry = True
            current_key = line.split("{", 1)[1].split(",", 1)[0].strip()
            temp_entries = [line]
        elif inside_entry:
            temp_entries.append(line)

        if inside_entry and line.strip() == "}":
            inside_entry = False
            if current_key in cite_keys:
                saved_entries.extend(temp_entries)
                saved_entries.append('\n')

    return saved_entries

# example usage
tex_filename = 'main.tex'
bib_filename = 'reference_full.bib'
cite_keys = extract_cite_content_from_file(tex_filename)
filtered_bib_entries = extract_bib_entries(bib_filename, cite_keys)

with open('reference.bib', 'w', encoding='utf-8') as out_file:
    out_file.writelines(filtered_bib_entries)

