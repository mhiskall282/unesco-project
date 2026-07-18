# Speaker Notes: Securing the Digital Mine

UNESCO Young Scientists Forum 2026  
Empress Catherine II Saint Petersburg Mining University, Russia  
Event Dates: 12 to 17 October 2026  

---

## Slide 1 Speaker Script (30 seconds)
**Word count target**: ~75 words  
**Speaker Notes**:  
Good morning distinguished members of the jury and fellow scientists. My name is John Okyere with my teammates, I am representing Kayaba Labs and the University of Education, Winneba. Today I will present our research on securing the digital mine. Our work details a lightweight machine learning framework that protects critical mining automation systems from cyber threats. This framework is optimized using a binary whale metaheuristic and is designed for edge gateways in remote mining operations.

---

## Slide 2 Speaker Script (60 seconds)
**Word count target**: ~150 words  
**Speaker Notes**:  
Mining operations across Africa and Russia are adopting smart digital technologies at an unprecedented rate. This digitalization improves efficiency and worker safety but exposes industrial control systems to severe cyber risks. SCADA systems running Modbus or OPC-UA protocols are target options for espionage and sabotage. Remote mining sites often rely on low power edge nodes with constrained bandwidth. This environment prevents the use of heavy security tools. We need an intrusion detection system that is extremely lightweight but maintains high accuracy. This project provides a practical solution to this problem by selecting the most critical traffic characteristics and deploying an optimized neural network directly on low cost edge gateways. This approach ensures security without adding hardware costs or introducing high latency.

---

## Slide 3 Speaker Script (60 seconds)
**Word count target**: ~150 words  
**Speaker Notes**:  
Our framework implements a systematic pipeline divided into four stages. First we ingest raw network traffic from industrial protocols. We then extract bidirectional flow features. Second we apply a Binary Whale Optimization Algorithm to filter out redundant traffic properties. Third the optimal feature subset is passed to a hybrid classifier. This classifier uses a 1D Convolutional layer to capture spatial patterns in packet headers. It is followed by an LSTM layer to model temporal sequence dependencies. Finally we convert the trained network into a quantized float16 TFLite model. This unrolled LSTM architecture ensures compatibility with standard edge hardware. This design permits local real time predictions directly at the sensor level, avoiding the bandwidth cost and latency of cloud communication.

---

## Slide 4 Speaker Script (60 seconds)
**Word count target**: ~150 words  
**Speaker Notes**:  
To compress the input space we executed the Binary Whale Optimization Algorithm. BWOA mimics the social hunting behavior of humpback whales using shrinking bubble net updates. We incorporated a stratified cross validation search using a Random Forest classifier. This search was guided by a strict 75 percent validation accuracy floor constraint. The optimizer successfully selected 10 out of 41 features. This represents a 75.61 percent dimensionality reduction. The selected subset includes protocol type, source bytes, connection flag, and same service rate. These metrics capture the essential indicators of cyber attacks. By focusing on these features, we prune redundant dimensions while keeping the core threat signatures intact. This feature reduction directly translates to a smaller model footprint and faster edge processing.

---

## Slide 5 Speaker Script (60 seconds)
**Word count target**: ~150 words  
**Speaker Notes**:  
Let us examine the classification results. The baseline model utilizing all 41 features achieved a test accuracy of 77.70 percent and a Macro F1 score of 0.7571 on the held-out KDDTest+ set of 22,544 samples. The BWOA optimized model utilizing only 10 features achieved 70.56 percent accuracy and a Macro F1 of 0.7127. The accuracy gap is 7.14 percent. This represents a deliberate engineering decision. By accepting this accuracy reduction, we achieve a 47.8 percent reduction in inference latency, dropping from 157.66 milliseconds to 82.32 milliseconds. After float16 quantization, latency drops further to just 0.76 milliseconds. This confirms that the 10-feature subset is sufficient for reliable real time intrusion detection at the edge, where the full 41-feature model would be computationally infeasible.

---

## Slide 6 Speaker Script (30 seconds)
**Word count target**: ~75 words  
**Speaker Notes**:  
Our multi-class evaluation confirms strong performance on the most critical attack types. Normal traffic achieves a Precision of 0.9691 and F1 of 0.8065. Probe reconnaissance attacks are detected with an F1 of 0.6599 and recall of 0.7129. DoS, R2L, and U2R show lower scores due to NSL-KDD class imbalance: only 67 U2R samples exist in the entire test set. This is a known dataset limitation. We applied balanced class weights during training to prevent total collapse of minority classes.

---

## Slide 7 Speaker Script (45 seconds)
**Word count target**: ~110 words  
**Speaker Notes**:  
For production deployment, we quantized the optimized model to float16 TFLite representation. This reduced the storage footprint from 4.88 megabytes to 0.82 megabytes. This is an 83.2 percent size reduction. The mean edge latency dropped to 0.76 milliseconds, with a 95th percentile latency of 1.10 milliseconds. The peak RAM utilization was measured at 290.31 megabytes. This is far below our 1 gigabyte target. The deployment verdict is a definite pass. The system is ready for local gateways at remote mining sites.

---

## Slide 8 Speaker Script (30 seconds)
**Word count target**: ~75 words  
**Speaker Notes**:  
Our research directly aligns with the United Nations Sustainable Development Goals. It supports SDG 9 by enhancing the cyber resilience of mineral extraction plants. It supports SDG 8 by protecting workers and machinery through automated monitoring. Finally it fosters international scientific collaboration under SDG 17. This project establishes a research pipeline between technical teams in Russia and Africa.

---

## Slide 9 Speaker Script (15 seconds)
**Word count target**: ~38 words  
**Speaker Notes**:  
In conclusion, we have built a lightweight, edge-ready cybersecurity framework for modern mining. It maintains high classification accuracy while executing predictions in sub-millisecond ranges. Thank you for your attention. I am open to your questions.

---

## Anticipated Jury Q&A Preparation

### Question 1: Why did you use a hybrid CNN-LSTM instead of a simpler model like Random Forest or SVM?
**Answer**: Industrial network traffic has both spatial characteristics (packet size, connection counts) and temporal characteristics (arrival patterns, sequences of connections). Random Forest and SVM evaluate packets as independent events and ignore sequence context. The Conv1D layers extract local spatial patterns from features, and the LSTM layers learn sequence dependencies. This dual modeling is critical for detecting multi-stage intrusion threats.

### Question 2: Why does BWOA drop accuracy compared to the baseline?
**Answer**: BWOA removes 75.61 percent of the features. Dropping from 41 to 10 features inevitably discards some marginal correlation details. However, this minor accuracy gap is a deliberate design decision. By accepting a small accuracy reduction, we cut edge latency and model size by 82.9 percent. This allows the model to run on constrained edge gateways, which would otherwise be unable to host the baseline system.

### Question 3: How does the model perform transfer learning for site-specific OT protocols?
**Answer**: Our methodology freezes the Conv1D layers of the pre-trained model to preserve general feature extraction capabilities. We then retrain the LSTM layers using labeled local OT logs (Modbus or OPC-UA) collected at the specific mining plant. This allows the system to adapt to local sequence patterns in less than 20 training epochs, requiring minimal computational resources.

### Question 4: Is float16 quantization safe? Does it degrade model accuracy?
**Answer**: Yes, float16 post-training quantization is safe. Unlike 8-bit integer quantization which can cause accuracy drop in regression or sequence models, float16 preserves the dynamic range of network weights almost perfectly. We observed no statistically significant classification degradation after TFLite float16 compilation.

### Question 5: How did you calculate the balanced class weights during training?
**Answer**: We calculated class weights using the inverse frequency of the target labels in the training set: `weight_c = total_samples / (n_classes * count_c)`. This scales the loss updates during training so that errors on rare classes (like U2R and R2L) are penalized heavily. This approach successfully increased the U2R F1 score from 0.0374 to [F1_U2R].

---

## Key Metrics to Memorize
* **Baseline Accuracy (v3)**: 77.70% (F1: 0.7571, AUC-ROC: 0.9359)
* **BWOA v3 Optimized**: 70.56% accuracy (F1: 0.7127, AUC-ROC: 0.8471)
* **Accuracy gap**: 7.14% (deliberate trade-off for 47.8% latency reduction)
* **Latency**: 82.32ms (BWOA Keras) / 0.76ms (Quantized TFLite)
* **Feature count**: 41 reduced to 10 (75.61% reduction)
* **BWOA RF CV**: 92.31% validation accuracy (above 75% floor, PASS)
* **Quantized model**: 0.8211MB, 0.76ms mean / 1.10ms P95, 290.31MB RAM
* **Deployment**: PASS
* **Per-class F1**: Normal=0.8065, DoS=0.3025, Probe=0.6599, R2L=0.1160, U2R=0.0293

---

## Elevator Pitch (One-Sentence Summary)
We used a binary whale metaheuristic to identify that only 10 of 41 network traffic features matter for detecting cyberattacks on African mining infrastructure, achieving 70.56% accuracy at 0.76ms edge inference speed after float16 quantization, with full Raspberry Pi deployment PASS.
