#preprocess.py
# 개인 포폴용 This is my personal project, so my code may not be perfect.
# note: this is for personal potofolio made by little ChatGPT and ME (Kwontaewook)

import os
import argparse
import json
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from scapy.all import rdpcap, TCP, UDP, IP

# flow 정의 (define flow)
def get_flow_id(pkt):
    if IP not in pkt:
        return None
    ip_layer = pkt[IP]
    proto = ip_layer.proto
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst


    #sprot, dport가 없을 경우 0으로 처리 ex) ICMP
    sport = pkt.sport if hasattr(pkt, 'sport') else 0
    dport = pkt.dport if hasattr(pkt, 'dport') else 0

    return (src_ip, dst_ip, sport, dport, proto)

# pcap 파일에서 flow 묶기 (from pcap file group by flow)
def parse_pcap(pcap_file):
    try:
        pkts = rdpcap(pcap_file)
    except Exception as e:
        print(f"파일 읽는 중 에러 발생 {pcap_file}: {e}")
        return {}
    
    flows = defaultdict(list)
    for p in pkts:
        if IP not in p:
            continue
        fid = get_flow_id(p)
        if fid is None:
            continue

        flows[fid].append((p.time, len(p), p))
    return flows


# flow 하나에 대한 특정치 뽑기  (extract features from a specific flow)
def summarize_flow(pkts_in_flows):
    if not pkts_in_flows:
        return None
    
    times = np.array(x[0] for x in pkts_in_flows)
    sizes = np.array(x[1] for x in pkts_in_flows)

    if len(times) == 0: # 위에서 걸리지만 혹시 몰라서.. (it should be never happen but just in case)
        return None
    
    duration = float(times.max() - times.min()) if len(times) > 1 else 0.0
    total_bytes = int(sizes.sum())
    pkt_cnt = int(len(sizes))

    mean_len = float(sizes.mean()) if pkt_cnt > 0 else 0.0
    std_len = float(sizes.std()) if pkt_cnt > 1 else 0.0

    iats = np.diff(times) if len(times) > 1 else np.array([0])
    mean_iat = float(iats.mean()) if len(iats) > 0 else 0.0
    std_iat = float(iats.std()) if len(iats) > 1 else 0.0


    # 길이 관련 min/max (just length related min/max)
    min_len = float(sizes.min())
    max_len = float(sizes.max())

    # TCP flag 카운트 (TCP flag count)
    syns = acks = fins = 0
    for _, _, pkt in pkts_in_flows:
        if TCP in pkt:
            flags = str(pkt[TCP].flags)
            if "S" in flags: syns += 1
            if "A" in flags : acks += 1
            if "F" in flags : fins += 1
        # elif UDP in pkt:
        #    UDP는 관련 플래그가 없음 (There is no realted flag in UDP)


    return {
        "duration": duration,
        "total_bytes": total_bytes,
        "pkt_count": pkt_cnt,
        "mean_len": mean_len,
        "std_len": std_len,
        "min_len": min_len,
        "max_len": max_len,
        "mean_iat": mean_iat,
        "std_iat": std_iat,
        "syn_count": syns,
        "ack_count": acks,
        "fin_count": fins
    } 

def main(pcap_dir, out_csv, label_map_file=None):
    all_records = []


    # 레이블 매핑 로드 (load label mapping)
    label_map = {}
    if label_map_file and os.path.exists(label_map_file):
        with open(label_map_file, 'r') as jf:
            label_map = json.load(jf)
    
    pcap_list = [os.path.join(pcap_dir, fn) for fn in os.listdir(pcap_dir)
                 if fn.endswith(".pcap") or fn.endswith(".pcapng")]
    
    for pcap_file in tqdm(sorted(pcap_list)):
        flow_dict = parse_pcap(pcap_file)
        label = label_map.get(os.path.basename(pcap_file), "unknown")

        for flow_id, pkts in flow_dict.items():
            stats = summarize_flow(pkts)
            if stats is None:
                continue
            

            src, dst, sport, dport, proto = flow_id
            stats.update({
                "src": src,
                "dst": dst,
                "sport": sport,
                "dport": dport,
                "proto": proto,
                "pcap_file": os.path.basename(pcap_file),
                "label": label
            })

            all_records.append(stats)

    df = pd.DataFrame(all_records)
    df.to_csv(out_csv, index=False)
    print(f"+ 총 {len(df)}개의 flow를 {out_csv}에 저장 완료")

    # parquet도 추가 저장
    parquet_path = out_csv.replace(".csv", ".parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"+ parquet 형식으로도 저장됨: {parquet_path}")

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--pcap_dir", type=str, required=True, help="pcap 파일들이 있는 폴더")
    argp.add_argument("--out-csv", type=str, required=True, help="출력 csv 파일 경로")
    argp.add_argument("--label-map", type=str, default="label map", required=False, help="json 레이블 파일")
    args = argp.parse_args()

    main(args.pcap_dir, args.out_csv, args.label_map)

    


