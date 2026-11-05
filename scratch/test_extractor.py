import os
import sys
import json
import glob

sys.path.insert(0, '/home/aicloud/extractive-summarization/src')
from summarizer.segmentation import split_sentences

def test_file(path):
    with open(path) as f:
        data = json.load(f)
    
    sentences = []
    uid = 0
    
    # 1. Overview
    es = data.get('executive_summary', {})
    narrative = es.get('narrative') if isinstance(es, dict) else ''
    overview = []
    if narrative:
        for s in split_sentences(narrative):
            if len(s.split()) >= 4:
                uid += 1
                overview.append({'uid': uid, 'role': 'overview', 'text': s})
    
    # 2. Findings & Recommendations
    findings = []
    recommendations = []
    obs_list = data.get('detailed_observations', [])
    if isinstance(obs_list, list):
        for item in obs_list:
            if not isinstance(item, dict):
                continue
            title = item.get('title', '')
            sev = item.get('severity', '')
            status = item.get('status', '')
            
            # desc & impact
            for field in ['observation_description', 'impact', 'likely_root_cause']:
                val = item.get(field, '')
                if isinstance(val, str) and val.strip():
                    for s in split_sentences(val):
                        if '[IMAGE_' in s or len(s.split()) < 4:
                            continue
                        uid += 1
                        findings.append({'uid': uid, 'role': 'findings', 'title': title, 'sev': sev, 'text': s})
            
            # recommendation
            rec = item.get('recommendation', '')
            if isinstance(rec, str) and rec.strip():
                for s in split_sentences(rec):
                    if len(s.split()) < 4:
                        continue
                    uid += 1
                    recommendations.append({'uid': uid, 'role': 'recommendations', 'title': title, 'sev': sev, 'text': s})
                    
    print(f"{os.path.basename(path)}: overview={len(overview)}, findings={len(findings)}, recs={len(recommendations)}")
    if overview:
        print("  Sample overview:", overview[0]['text'][:80])
    if findings:
        print("  Sample finding:", findings[0]['text'][:80])
    if recommendations:
        print("  Sample rec:", recommendations[0]['text'][:80])

for f in ['1.json', '11.json', '24.json', '36.json', '55.json']:
    test_file(f'data2/{f}')
