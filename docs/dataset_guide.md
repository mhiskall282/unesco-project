# Dataset Ingestion and Preprocessing Guide

This document details the configuration, feature listings, acquisition steps, and characteristics of the datasets used to train and validate our framework.

---

## 1. NSL-KDD Dataset
An improved, cleaner version of the classic KDD Cup 1999 dataset. It resolves structural issues such as duplicate records and biases towards specific attack classes.

### Access and Ingestion
The raw files `KDDTrain+.txt` and `KDDTest+.txt` are automatically downloaded from public repositories upon calling `NSLKDDLoader.load()`. Files are saved under the `data/raw/` directory.

### Target Attack Classes (5-Class)
All network traffic labels are mapped into five categories:
- **Normal**: Standard operational network requests.
- **DoS (Denial of Service)**: Attempts to overwhelm host resources (e.g., neptune, smurf, back).
- **Probe**: Reconnaissance attacks seeking port openings and configurations (e.g., satan, ipsweep, nmap).
- **R2L (Remote-to-Local)**: Unauthorized local access from remote systems (e.g., guess_passwd, warezmaster).
- **U2R (User-to-Root)**: Attempts to gain administrator privileges (e.g., buffer_overflow, rootkit).

### Full Feature Listing (41 features)
1. `duration` (continuous)
2. `protocol_type` (categorical: tcp, udp, icmp)
3. `service` (categorical: http, ftp, smtp, etc.)
4. `flag` (categorical: SF, S0, REJ, etc.)
5. `src_bytes` (continuous)
6. `dst_bytes` (continuous)
7. `land` (binary: 0 or 1)
8. `wrong_fragment` (continuous)
9. `urgent` (continuous)
10. `hot` (continuous)
11. `num_failed_logins` (continuous)
12. `logged_in` (binary: 0 or 1)
13. `num_compromised` (continuous)
14. `root_shell` (binary: 0 or 1)
15. `su_attempted` (binary/continuous)
16. `num_root` (continuous)
17. `num_file_creations` (continuous)
18. `num_shells` (continuous)
19. `num_access_files` (continuous)
20. `num_outbound_cmds` (continuous)
21. `is_host_login` (binary: 0 or 1)
22. `is_guest_login` (binary: 0 or 1)
23. `count` (continuous)
24. `srv_count` (continuous)
25. `serror_rate` (continuous)
26. `srv_serror_rate` (continuous)
27. `rerror_rate` (continuous)
28. `srv_rerror_rate` (continuous)
29. `same_srv_rate` (continuous)
30. `diff_srv_rate` (continuous)
31. `srv_diff_host_rate` (continuous)
32. `dst_host_count` (continuous)
33. `dst_host_srv_count` (continuous)
34. `dst_host_same_srv_rate` (continuous)
35. `dst_host_diff_srv_rate` (continuous)
36. `dst_host_same_src_port_rate` (continuous)
37. `dst_host_srv_diff_host_rate` (continuous)
38. `dst_host_serror_rate` (continuous)
39. `dst_host_srv_serror_rate` (continuous)
40. `dst_host_rerror_rate` (continuous)
41. `dst_host_srv_rerror_rate` (continuous)

### Limitations
NSL-KDD contains dated cyberattack methods and lacks distinct industrial protocol formats such as Modbus, DNP3, or OPC-UA. Hence, it is strictly used as our baseline validation target.

---

## 2. SWaT Dataset
The Secure Water Treatment (SWaT) dataset is a real-world testbed representing continuous water filtration processes under physical cyberattacks.

### Access Instructions
Due to security controls, researchers must request access directly from the iTrust Centre for Research in Cyber Security at Singapore University of Technology and Design (SUTD).
Web address: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

### Attack Scenarios and Features
Contains 51 features representing industrial sensors (flow, pressure, pH level) and actuators (valves, pumps). Attacks focus on sensor spoofing, physical state alterations, and Denial of Service on PLCs.

---

## 3. BATADAL Dataset
The Battle of the Attack Detection Algorithms (BATADAL) dataset focuses on cyberattack detection within water distribution networks.

### Access Instructions
Files can be downloaded directly from the official challenge site at http://www.batadal.net or public repositories.

### Attack Scenarios and Features
Consists of 43 features detailing water tank levels, pump speeds, and flow rates. The labels indicate attack status (1: attack, 0: normal). Attacks typically involve SCADA spoofing, pump command overrides, and reservoir overflows.

---

## 4. Custom OT Dataset (Phase 1)
Planned data capture directly from pilot subsoil plants and mineral operations.

### Capture Architecture
We deploy cloud-based AWS EC2 sniffer nodes connected to network taps at pilot mine processing facilities. This infrastructure records active control traffic.

### Target Protocols
- **Modbus/TCP**: Master/slave polling registers for physical actuators.
- **OPC-UA**: Server/client data channels for digital twins and SCADA systems.
- **DNP3**: Wide-area substation telemetry.

### Labeling Strategy
Labels are assigned based on anomalous physical mine state shifts, unexpected IP addresses, and simulated payload modifications. Features are converted to flow statistics using CICFlowMeter, then mapped to 41 NSL-KDD equivalents for transfer learning evaluation.
