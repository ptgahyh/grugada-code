#!/usr/bin/env python3
"""
Download one skill from a trusted GitHub source, clean it, scan it, and install or zip it.

Usage:
    python fetch_skill.py <name> [--source kdense|anthropic] [--dest DIR] [--zip OUT.zip] [--force] [--drop FILE]

- Downloads SKILL.md plus every references/, scripts/, assets/, templates/ file it links to
  (recursively through linked .md files) from raw.githubusercontent.com.
- Removes the K-Dense "Citing Scientific Agent Skills" section (it tells Claude to add the
  vendor's paper to the user's manuscript).
- Scans for risky patterns. If anything is found it stops (exit 2) without installing or
  zipping; rerun with --force (optionally --drop the risky file) only after the user agrees.
- Validates frontmatter (name must equal the folder name; YAML must parse).
Standard library only (PyYAML used if available).
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

SOURCES = {
    'kdense': 'https://raw.githubusercontent.com/K-Dense-AI/scientific-agent-skills/main/skills',
    'anthropic': 'https://raw.githubusercontent.com/anthropics/skills/main/skills',
}
LINK = re.compile(r'(?<![\w/.-])((?:references|scripts|assets|templates)/[A-Za-z0-9_./-]+'
                  r'\.(?:md|py|json|txt|csv|yaml|yml|tex|sh|js|ts|html|css))')
RISKY = {
    'paid/third-party AI API': re.compile(r'openrouter|api\.openai\.com|replicate\.com|OPENAI_API_KEY', re.I),
    'reads secrets/.env': re.compile(r'\.env\b|~/\.ssh|id_rsa|AWS_SECRET|credentials\.json', re.I),
    'shell download-and-run': re.compile(r'curl[^\n|]*\|\s*(ba)?sh|wget[^\n|]*\|\s*(ba)?sh'),
    'dynamic code exec': re.compile(r'(?<![.\w])(eval|exec)\(|os\.system\('),
    'hidden-instruction wording': re.compile(r'ignore (all |any )?previous|do not (tell|inform) the user|without telling', re.I),
    'vendor upsell': re.compile(r'k-dense web|paid plan', re.I),
}
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def download(name, source, out):
    base = f'{SOURCES[source]}/{name}'
    todo, seen = ['SKILL.md'], set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        try:
            data = fetch(f'{base}/{rel}')
        except OSError:
            if rel == 'SKILL.md':
                sys.exit(f'Không tìm thấy skill "{name}" ở nguồn {source}.')
            print(f'  (bỏ qua, không tải được: {rel})')
            continue
        seen.add(rel)
        path = os.path.join(out, name, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)
        if rel.endswith('.md'):
            todo.extend(LINK.findall(data.decode('utf-8', 'ignore')))
    return sorted(seen)


def clean(skill_md):
    text = open(skill_md, encoding='utf-8').read()
    cut = text.find('## Citing Scientific Agent Skills')
    if cut > 0:
        text = text[:cut].rstrip() + '\n'
        open(skill_md, 'w', encoding='utf-8').write(text)
        print('  Đã bỏ mục "Citing Scientific Agent Skills".')
    return text


def validate(text, name):
    parts = text.split('---')
    if len(parts) < 3:
        return ['thiếu frontmatter ---']
    errors = []
    try:
        import yaml
        fm = yaml.safe_load(parts[1])
    except ImportError:
        fm = dict(re.findall(r'^(name|description):\s*(.*)$', parts[1], re.M))
    except Exception as e:  # noqa: BLE001 - report any YAML error to the caller
        return [f'YAML lỗi: {e}']
    if fm.get('name') != name:
        errors.append(f'name "{fm.get("name")}" khác tên thư mục "{name}"')
    if not NAME_RE.match(name):
        errors.append('tên skill có ký tự không hợp lệ')
    if not fm.get('description'):
        errors.append('thiếu description')
    return errors


def scan(root):
    """Gộp cảnh báo theo (loại, file) -> danh sách dòng, để báo cáo ngắn gọn."""
    hits = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            p = os.path.join(dirpath, fn)
            text = open(p, encoding='utf-8', errors='ignore').read()
            for label, pat in RISKY.items():
                lines = sorted({text.count('\n', 0, m.start()) + 1 for m in pat.finditer(text)})
                if lines:
                    shown = ','.join(map(str, lines[:8])) + ('…' if len(lines) > 8 else '')
                    hits.append(f'{label}: {os.path.relpath(p, root)} dòng {shown}')
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--source', default='kdense', choices=SOURCES)
    ap.add_argument('--dest', help='thư mục skills để cài vào (vd .claude/skills hoặc ~/.claude/skills)')
    ap.add_argument('--zip', help='đường dẫn file ZIP để tải lên claude.ai')
    ap.add_argument('--force', action='store_true', help='vẫn cài/zip dù có cảnh báo (chỉ khi người dùng đồng ý)')
    ap.add_argument('--drop', action='append', default=[], help='bỏ một file (đường dẫn tương đối) trước khi cài')
    args = ap.parse_args()
    if not NAME_RE.match(args.name):
        sys.exit('Tên skill không hợp lệ.')

    tmp = tempfile.mkdtemp(prefix='skill-')
    files = download(args.name, args.source, tmp)
    root = os.path.join(tmp, args.name)
    text = clean(os.path.join(root, 'SKILL.md'))
    size_kb = sum(os.path.getsize(os.path.join(root, f)) for f in files) // 1024
    print(f'Tải {len(files)} file, {size_kb} KB từ {args.source}/{args.name}')

    errors = validate(text, args.name)
    hits = scan(root)
    for e in errors:
        print(f'  LỖI định dạng: {e}')
    for h in hits:
        print(f'  CẢNH BÁO {h}')
    if errors:
        sys.exit(1)

    if hits and not args.force:
        print(f'DỪNG: có cảnh báo, chưa cài/zip. Bản tải về ở {root}. Cho người dùng xem cảnh báo; '
              'chỉ chạy lại với --force (và --drop <file> để bỏ file rủi ro) khi họ đồng ý.')
        sys.exit(2)
    for rel in args.drop:
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            os.remove(path)
            files = [f for f in files if f != rel]
            print(f'  Đã bỏ file {rel}')

    final = root
    if args.dest:
        final = os.path.join(os.path.expanduser(args.dest), args.name)
        if os.path.exists(final):
            shutil.rmtree(final)
        shutil.copytree(root, final)
        print(f'Đã cài vào {final}')
    if args.zip:
        with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(os.path.join(root, f), os.path.join(args.name, f))
        print(f'ZIP để tải lên claude.ai: {args.zip}')
    print(f'Đọc ngay để dùng trong phiên này: {os.path.join(final, "SKILL.md")}')

if __name__ == '__main__':
    main()
