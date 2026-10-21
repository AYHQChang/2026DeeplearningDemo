"""教师离线准备：python prepare_data.py /path/to/chnsenticorp.tar.gz。

只接受 README 指定的官方有标签旧版；不下载，不生成评论，不修改标签。
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import io
import json
import random
import re
import tarfile
import unicodedata

SOURCE_URL = 'https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz'
SOURCE_SHA256 = '7251a6c7bedf7116b4727c809a694405187e1e9cbdb770f509504b16ca6d9c53'
SPLIT_MD5 = {'train': '689360c4a4a9ce8d8719ed500ae80907',
             'dev': '05e4b02561c2a327833e05bbe8156cec',
             'test': '917dfc6fbce596bb01a91abaa6c86f9e'}
VERSION = 'chnsenticorp-paddlehub-v2.0.0-unique-course-v1'


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def prepare(archive_path, destination):
    raw = Path(archive_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('SHA256不符合固定官方版本；不能混用同名数据包。')
    by, source_files = {}, {}
    groups = collections.defaultdict(list)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for split in ('train', 'dev', 'test'):
            content = archive.extractfile(f'chnsenticorp/{split}.tsv').read()
            if hashlib.md5(content).hexdigest() != SPLIT_MD5[split]:
                raise ValueError(f'{split} 校验值不符。')
            source_files[split] = {'md5': SPLIT_MD5[split], 'sha256': hashlib.sha256(content).hexdigest()}
            lines = content.decode('utf8').splitlines()
            if lines[0] != 'label\ttext_a':
                raise ValueError('需要带真实 label 的原始划分。')
            by[split] = []
            for line_number, line in enumerate(lines[1:], start=2):
                label, text = line.split('\t', 1)
                if label not in ('0', '1') or not normalize(text):
                    raise ValueError(f'非法标签或空文本 {split}:{line_number}')
                row = {'id': f'{split}:{line_number}', 'text': text, 'label': int(label)}
                by[split].append(row)
                groups[normalize(text)].append((split, row))
    duplicates = {text: rows for text, rows in groups.items() if len(rows) > 1}
    # 按预先声明的文本一致性规则排除全部重复出现，不按难度或模型表现挑选。
    eligible = {s: [r for r in rows if len(groups[normalize(r['text'])]) == 1] for s, rows in by.items()}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for index, split in enumerate(('train', 'dev', 'test')):
        sample = []
        for label in (0, 1):
            candidates = [r for r in eligible[split] if r['label'] == label]
            sample.extend(random.Random(42 + 100 * index + label).sample(candidates, 1000 if split == 'train' else 250))
        random.Random(1042 + index).shuffle(sample)
        name = 'val' if split == 'dev' else split
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=['id', 'text', 'label'], delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(sample)
        content = buffer.getvalue().encode('utf8')
        (destination / f'{name}.tsv').write_bytes(content)
        files[name] = {'file': f'{name}.tsv', 'source_split': split, 'count': len(sample),
                       'label_counts': dict(collections.Counter(str(r['label']) for r in sample)),
                       'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    manifest = {
        'version': VERSION, 'class_names': ['负面', '正面'], 'label_mapping': {'0': '负面', '1': '正面'},
        'source': {'url': SOURCE_URL, 'archive_md5': hashlib.md5(raw).hexdigest(), 'archive_sha256': SOURCE_SHA256,
                   'loader': 'https://github.com/PaddlePaddle/PaddleNLP/blob/v2.0.0/paddlenlp/datasets/chnsenticorp.py',
                   'files': source_files, 'download_verified_date': '2026-10-17',
                   'note': '仅使用该tar.gz有标签版，未混入当前无标签test的zip版。'},
        'raw_counts': {s: len(rows) for s, rows in by.items()},
        'audit': {'normalization': 'nfkc-whitespace-v1', 'duplicate_text_groups': len(duplicates),
                  'cross_split_text_groups': sum(len({s for s, r in rows}) > 1 for rows in duplicates.values()),
                  'conflicting_label_groups': sum(len({r['label'] for s, r in rows}) > 1 for rows in duplicates.values()),
                  'excluded_rows': {s: len(by[s]) - len(eligible[s]) for s in by},
                  'eligible_counts': {s: len(rows) for s, rows in eligible.items()},
                  'rule': '任意规范化重复文本的全部出现均排除；不迁移划分、不改标签；仅排除完全一致文本，未检查语义近重复。',
                  'course_cross_split_overlap': 0},
        'sampling': {'seed': 42, 'stratified_by_label': True, 'preserve_source_split': True,
                     'per_class': 'train=1000/dev=250/test=250',
                     'algorithm': 'random.Random(42+100*split_index+label).sample; shuffle seed=1042+split_index',
                     'split_order': ['train', 'dev', 'test']},
        'preprocessing': {'normalization': 'nfkc-whitespace-v1', 'character_level': True,
                          'pad_id': 0, 'unk_id': 1, 'vocabulary_source': 'course train only', 'max_length_default': 100},
        'files': files,
        'use_and_redistribution': {
            'pinned_archive_has_license_file': False, 'code_license_is_not_data_license': True,
            'local_use': '仅本地非商业课程教学，不声明数据为Apache/MIT。',
            'related_luge_terms': '另一个当前LUGE zip包附Participant Agreement，2.2允许非商业科研/教学；不将其当作固定旧包明确公共再分发授权。',
            'public_redistribution': '未核实到固定旧包明确公共再分发授权；原始包/评论TSV/课程zip已gitignored，公开发布前须另行核实。'},
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    result = prepare(args.archive, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
