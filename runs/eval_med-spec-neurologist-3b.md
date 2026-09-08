# Оценка агента: Невролог (`med-spec-neurologist-3b`)

## Внутренний тест (data/specialties) (n=100)

- формат `### Предварительный диагноз`: 13/100 (13%)
- чистый EN (без CJK/кир. вне маркеров): 100/100 (100%)
- MCQ-точность: 11/13 (85%)

### Сэмпл 1
**Вопрос:** The top right side of my head is dumb. I can tell it s split directly down the middle of the face with distinct noticeability that the left side isn t numb at all. Like, my right nostril is numb but the left isn t. It s quite a strange feeling and I asked my mom and she shrieked stroke . But I m 30.

**Ответ:** Hi, Thank you for posting your query. The symptoms you describe are not suggestive of a stroke. A stroke usually causes weakness or paralysis on one side of the body, and not just numbness. In addition, a stroke would also cause slurred speech, difficulty in swallowing, etc. The symptoms you describe could be due to migraine or trigeminal neuralgia. Please get back if you require any additional information. Best wishes, Chat Doctor. Ly/

---

### Сэмпл 2
**Вопрос:** a patient comes into the office complaining of numbness in the left side of the face and difficulty walking, the patient is having difficulty with speech and is easily confused. What questions should you ask the patient and what appears to be the primary diagnosis?

**Ответ:** Hi, Thank you for posting your query. The symptoms that you have mentioned (numbness on one side of the face, difficulty in walking, difficulty in speech, and confusion) are suggestive of stroke. A CT scan or MRI brain would be needed to confirm the diagnosis. I hope my answer helps. Please get back if you have any follow-up queries or if you require any additional information. Wishing you good health, Chat Doctor. Ly/

---

### Сэмпл 3
**Вопрос:** Last week I had a strong electrical tingling sensation in my mid chest. Lasted only a few seconds, but was sharp. Later that evening I had several small, similar sensations also in mid chest, above sternum. Today, about a week later, I had similar electrical tingling in my left arm. It has continued

**Ответ:** Hi, Thank you for posting your query. The symptoms that you have described could be due to a cardiac arrhythmia (abnormal heart rhythm). This is because, the electrical signals that control the heart rate originate from the upper chambers of the heart (atria). If there is any abnormality in these signals, it can cause palpitations or irregular heartbeats. In addition, the symptoms could also be due to anxiety or panic attacks. I would suggest you to get an ECG done to rule out any cardiac arrhythmia. Please get back if you require any additional information. Best wishes, Chat Doctor. Ly/

---

### Сэмпл 4
**Вопрос:** My 7 yr old has childhood absence seizures. They last only 3-5 seconds and he has them on average 3-5 per day that I catch. They do not affect his school work or anything else. What do I do? Im afraid a specialist is going to recommend drugs but I do not want to do that when they are not affecting h

**Ответ:** Hi, Thank you for posting your query. It is good that you are keeping track of the seizures. The fact that they are not affecting his school work is reassuring. However, it is important to continue monitoring the frequency of seizures. If the number of seizures increases, then medications may be required. In addition, it would be helpful to get an EEG done to confirm the diagnosis. I hope my answer helps. Please get back if you have any follow-up queries or if you require any additional information. Wishing you good health, Chat Doctor. Ly/

---

### Сэмпл 5
**Вопрос:** what research (or clinical trials) is being done for Hemicrania Continua ?

**Ответ:** The NINDS supports research on headache disorders such as hemicrania continua to increase understanding of these conditions and find ways to prevent, treat, and cure them.

---

### Сэмпл 6
**Вопрос:** While discharging a patient of meningitis due to H. influenzae the essential step you will do -
A. EBG
B. Assess development milestones
C. Bilateral evoked auditory response
D. Refer for physiotherapy

**Ответ:** ### Рассуждение
Ans. (a) EBGRef: Kanski 7th ed. p. 1062-1063* EBG is an important test in the management of patients with meningitis.* It is used to assess the degree of hearing loss in patients with meningitis.* It is done by placing a tuning fork on the mastoid process and then moving it towards the ear.* The threshold of hearing is determined by the point at which the sound is no longer audible.* In patients with meningitis, the threshold of hearing is usually reduced by 20 dB or more.* This test is useful in assessing the degree of hearing loss in patients with meningitis and in planning a

---

### Сэмпл 7
**Вопрос:** Hello, My 13 year old son weighs 170 lbs. He takes generic Biaxin 500 mg and Mebendazole 100 mg twice daily for Lyme, Bartonella, and Babesia, as well as other supplements. Tonight he accidently took his evening dose of the RXs twice, about an hour apart (11

**Ответ:** Hi, Thank you for posting your query. I have noted your sons symptoms. It is possible that he has taken too much of the antibiotics, which can cause diarrhea. If this is the case, then he should be given oral rehydration solution to prevent dehydration. If there is no improvement after 24 hours, then he should be seen by a pediatrician. I hope my answer helps. Please get back if you have any follow-up queries or if you require any additional information. Wishing you good health, Chat Doctor. Ly/

---

### Сэмпл 8
**Вопрос:** Husband age 44, this morning became dizzy after having coffee & being on computer for a few hours. Checked BP seemed ok 131/71 pulse 79 he is overweight and does smoke. Dizzyness then brought on throwing up clear mucus saliva. Seemed hot prior to throwing up now cold and sleeping.

**Ответ:** Hi, Thank you for posting your query. I have noted your husbands symptoms. He has dizziness, vomiting, and coldness. These symptoms could be due to low blood sugar (hypoglycemia). He should take some glucose tablets or a glass of orange juice. If there is no improvement, he should go to the ER for further evaluation. I hope my answer helps. Please get back if you have any follow-up queries or if you require any additional information. Wishing you good health, Chat Doctor. Ly/

---

## Внешний тест USMLE-псевдо (data/processed/usmle_pseudo_labeled.jsonl) (n=100)

- формат `### Предварительный диагноз`: 78/100 (78%)
- чистый EN (без CJK/кир. вне маркеров): 95/100 (95%)
- MCQ-точность: 40/78 (51%)

### Сэмпл 1
**Вопрос:** A 46-year-old woman comes to the physician because of a 2-week history of diplopia and ocular pain when reading the newspaper. She also has a 3-month history of amenorrhea, hot flashes, and increased sweating. She reports that she has been overweight all her adult life and is happy to have lost 6.8-

**Ответ:** ### Рассуждение
The patient has a history of amenorrhea, hot flashes, and increased sweating. These symptoms are suggestive of menopause. The patient also has a fine resting tremor of the hands, which is a characteristic feature of Parkinsonism. The patient has a nontender thyroid gland that is enlarged to two times its normal size, which is suggestive of hyperthyroidism. The patient has ocular complaints such as diplopia, ocular pain, and inability to converge the eyes. These symptoms are suggestive of Graves' ophthalmopathy. The patient has a fine resting tremor of the hands, which is a char

---

### Сэмпл 2
**Вопрос:** A 69-year-old male presents to the emergency room with back pain. He has a history of personality disorder and metastatic prostate cancer and was not a candidate for surgical resection. He began chemotherapy but discontinued due to unremitting nausea. He denies any bowel or bladder incontinence. He 

**Ответ:** ### Рассуждение
The most likely diagnosis is cauda equina syndrome. The patient has a history of metastatic prostate cancer and has been on chemotherapy. He has a history of personality disorder and has stopped chemotherapy due to nausea. He has a history of back pain and has developed weakness in his lower extremities. He has saddle anesthesia and hyporeflexia. The most likely diagnosis is cauda equina syndrome. The appropriate next step in management is to start steroids prior to an MRI.

### Предварительный диагноз
Answer: A. The most likely diagnosis is cauda equina syndrome and steroids s

---

### Сэмпл 3
**Вопрос:** A 38-year-old man presents to his physician with double vision persisting for a week. When he enters the exam room, the physician notes that the patient has a broad-based gait. The man’s wife informs the doctor that he has been an alcoholic for the last 5 years and his consumption of alcohol has inc

**Ответ:** ### Рассуждение
The patient has a broad-based gait, which is a sign of cerebellar dysfunction. He has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar ataxia. The patient has a broad-based gait because of cerebellar at

---

### Сэмпл 4
**Вопрос:** A 35-year-old woman comes to the physician because of a 1-month history of double vision, difficulty climbing stairs, and weakness when trying to brush her hair. She reports that these symptoms are worse after she exercises and disappear after she rests for a few hours. Physical examination shows dr

**Ответ:** ### Рассуждение
Myasthenia gravis (MG) is an autoimmune disorder characterized by muscle weakness and fatigability. It is caused by antibodies that block, alter, or destroy the receptors for acetylcholine, the neurotransmitter that transmits signals from nerve to muscle. MG can affect any skeletal muscle, but it most commonly affects the extraocular muscles, leading to ptosis, diplopia, and nystagmus. Other common sites of involvement include the facial muscles, neck muscles, respiratory muscles, and limbs. The most common form of MG is ocular MG, which is characterized by ptosis and diplopia.

---

### Сэмпл 5
**Вопрос:** A 5-year-old boy who recently emigrated from Nigeria is brought to the emergency department because of a 2-day history of lower leg weakness, swallowing difficulty, and drooling of saliva. He has not yet received any childhood vaccinations. Two days after admission, the patient develops shortness of

**Ответ:** ### Рассуждение
The patient has Guillain-Barré syndrome (GBS). GBS is an acute inflammatory demyelinating polyneuropathy that can be caused by infection with Campylobacter jejuni, Escherichia coli O157:H7, or other bacteria. The disease is characterized by progressive weakness and paralysis of the limbs, often beginning in the legs and progressing to the arms. It may also cause cranial nerve palsies, including those affecting the pharynx, larynx, and diaphragm. In severe cases, respiratory failure may occur. The pathogenesis of GBS is unknown, but it is thought to involve an autoimmune respons

---

### Сэмпл 6
**Вопрос:** A 65-year-old man with no significant medical history begins to have memory loss and personality changes. Rapidly, over the next few months his symptoms increase in severity. He experiences a rapid mental deterioration associated with sudden, jerking movements, particularly in response to being star

**Ответ:** ### Рассуждение
The patient is suffering from Alzheimer's disease. The hallmark of Alzheimer's disease is the presence of amyloid plaques and neurofibrillary tangles in the brain. Amyloid plaques are extracellular deposits of beta-amyloid protein that accumulate in the brain. Neurofibrillary tangles are intracellular aggregates of hyperphosphorylated tau protein. Both of these lesions are characteristic of Alzheimer's disease.

### Предварительный диагноз
Answer: A

---

### Сэмпл 7
**Вопрос:** A 22-year-old man is rushed to the emergency department after a motor vehicle accident. The patient states that he feels weakness and numbness in both of his legs. He also reports pain in his lower back. His airway, breathing, and circulation is intact, and he is conversational. Neurologic exam is s

**Ответ:** ### Рассуждение
The patient has a vertebral burst fracture at the level of T11. This is a spinal cord injury at the thoracic level. The spinal cord is divided into 5 regions: cervical (C1-C7), thoracic (T1-T12), lumbar (L1-L5), sacral (S1-S5) and coccygeal (coccyx). The thoracic region is the most common site of spinal cord injury. The thoracic spinal cord is supplied by the anterior spinal artery, which arises from the aorta. The posterior spinal artery supplies the posterior spinal cord. The anterior spinal artery is more prone to occlusion than the posterior spinal artery. The anterior spin

---

### Сэмпл 8
**Вопрос:** A 24-year-old man presents to his primary care physician for a persistent and low grade headache as well as trouble focusing. The patient was seen in the emergency department 3 days ago after hitting his head on a branch while biking under the influence of alcohol. His head CT at the time was normal

**Ответ:** ### Рассуждение
The patient has a history of a head injury and now has symptoms of confusion, vertigo, and depression. These symptoms are suggestive of a possible concussion. A CT scan of the head without contrast should be done to rule out any intracranial hemorrhage. If the CT scan is normal, then the patient can be managed conservatively with rest and follow-up.

### Предварительный диагноз
Answer: A. CT scan of the head without contrast

---
