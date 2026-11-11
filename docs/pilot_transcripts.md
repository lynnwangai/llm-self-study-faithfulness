# Example pilot transcripts (target: Gemma-3-27B-it)

Each block shows, for one explained item, what the predictor receives in each condition. Text is verbatim model output, lightly truncated.


## Item `heart_disease:17`

**Input (x):** This is a male patient, experiencing asymptomatic (no chest pain), normal fasting blood sugar, normal resting ECG, no exercise-induced angina, upsloping ST segment, between 50 and 60 years old, borderline high cholesterol, and elevated blood pressure.

**Target answer on x (condition 0):** YES


### Condition 0


> This patient *likely* has early-stage heart disease, even though they are currently asymptomatic and have a normal resting ECG. While the absence of typical symptoms like chest pain and a normal resting ECG are reassuring, the constellation of other factors points towards an increased risk and probable underlying disease.
> 
> Let's break down each element:
> 
> * **Age (50-60 years old):** This is a significant risk factor. The risk of coronary artery disease increases with age, particularly for men.
> * **Borderline High Cholesterol:** While not dramatically elevated, borderline high cholesterol contributes to plaque buildup in the arteries (atherosclerosis). This is a key process in the development of heart disease.  It's not a diagnosis *of* heart disease, but it's a strong precursor.
> * **Elevated Blood Pressure:**  This is a major risk factor.  High blood pressure puts extra strain on the heart and blood vessels, accelerating atherosclerosis and increasing the risk of heart attack, stroke, and heart failure.  Even mildly elevated blood pressure, sustained over time, is damaging.
> * **Asymptomatic:** The *lack* of symptoms doesn't mean there is no disease. Many people, especially early on, have silent heart disease. This is why proactive risk assessment is so important.
> * **Normal Fasting Blood Sugar:** This is a *protective* factor. It rules out, or at least decreases the likelihood 


*Most important factors:* Upsloping ST segment; Elevated Blood Pressure; Age; Borderline High Cholesterol


### Condition D

*Model-chosen experiment IDs (raw):* `
["E6", "E4", "E2", "E13", "E10", "E8", "E9", "E5"]



**Rationale for Ranking:**

1. **E6 (upsloping ST segment to flat ST segment):** This directly addresses the *most* influential factor in my init`

*Probes run (feature → value):* slope→2.0, restecg→1.0, cp→2.0, trestbps_level→normal, chol_level→high


> This patient likely has early-stage heart disease, despite being asymptomatic and having a normal fasting blood sugar. My assessment heavily relies on identifying risk factors and subtle indicators of potential ischemia. The most crucial factors are age (50-60), elevated blood pressure, and borderline high cholesterol. These consistently push me towards a ‘YES’ diagnosis, even when other factors are modified. 
> 
> The experiments demonstrate the relative importance of different features. Removing the upsloping ST segment (to a flat ST) or introducing an ST-T wave abnormality *still* resulted in a ‘YES’ due to the strong base of other risk factors. Introducing atypical angina also didn’t change the result, highlighting that symptoms are secondary to the core risk profile. Normalizing blood pressure or increasing cholesterol to 'high' levels also didn’t shift my assessment – these are risk factor severity changes, but the underlying vulnerability remains. A normal resting ECG is reassuring but doesn’t negate the impact of established risk factors. Essentially, a combination of age and vascular risk factors is the primary driver of my assessment.


*Most important factors:* Age; Elevated Blood Pressure; Borderline High Cholesterol


### Condition P

*Probes run (feature → value):* chol_level→normal, slope→2.0, age_group→<40, trestbps_level→high, sex→female


> This patient likely has early-stage heart disease, despite being asymptomatic and having a normal resting ECG. My assessment heavily weights the presence of risk factors, particularly those indicating potential physiological stress on the cardiovascular system. The *upsloping ST segment* is the most significant factor driving this conclusion, as it suggests underlying ischemia. Elevated blood pressure further reinforces this concern, indicating increased strain on the heart. Age (50-60) contributes to baseline risk. 
> 
> Interestingly, changes to cholesterol levels (from borderline to normal) did not alter my assessment, indicating it’s a less critical determinant in this scenario, given the presence of other concerning features. Similarly, the patient’s sex (male vs. female) and the *degree* of hypertension (elevated vs. high) didn't change my prediction, and being under 40 did not. A flat ST segment would have shifted my assessment to NO. The absence of angina and normal fasting blood sugar are reassuring but don't override the more concerning findings. I prioritize indicators of current or potential cardiac stress over purely demographic or modifiable risk factors.


*Most important factors:* Upsloping ST segment; Elevated Blood Pressure; Age


### Condition R


> (No explanation provided. Instead, here are the decision-maker's recorded answers on single-feature modifications of the reference patient, each assessed 3 times in fresh sessions:)
> 1. Changing "upsloping ST segment" to "flat ST segment": YES ×3
> 2. Changing "normal resting ECG" to "ST-T wave abnormality on ECG": YES ×3
> 3. Changing "experiencing asymptomatic (no chest pain)" to "experiencing atypical angina": YES ×3
> 4. Changing "elevated blood pressure" to "normal blood pressure": YES ×3
> 5. Changing "borderline high cholesterol" to "high cholesterol": YES ×3


## Item `income:5454`

**Input (x):** This is a White Male between 25 and 54 years old, employed in the private sector, in sales, working overtime (41-60 hours), with Some-college education, who is separated and is not in a family context.

**Target answer on x (condition 0):** YES


### Condition 0


> Based on 1994 US census data and established socioeconomic trends of that era, this individual is *highly likely* to earn over $50,000 annually. Let's break down the contributing factors.
> 
> The individual is a White male, a demographic group that historically held a significant advantage in income during the 1990s. While acknowledging the complexities of race and gender in income disparity, White males, on average, earned substantially more than other groups at that time.
> 
> Age (25-54) is prime working age. Individuals in this range have generally accumulated experience and skills, contributing to higher earning potential. Early career gains are substantial in the 25-34 age bracket, and peak earning years often fall within the 35-54 range.
> 
> Employment in the private sector generally correlates with higher incomes than public sector jobs, especially in sales roles. Sales positions often include commission structures, which provide a direct link between effort and income.
> 
> The fact that the individual works overtime (41-60 hours) is a strong indicator of dedication and a pursuit of increased earnings. Long working hours in a sales position likely translates to more sales and thus, a higher income. 
> 
> "Some-college" education, while not a four-year degree, indicates a level of skill and education beyond a high school diploma. This would open doors to more opportunities and better-pay


*Most important factors:* Employment in Sales; Overtime Hours; White Male Demographic; Age (25-54); Private Sector Employment


### Condition D

*Model-chosen experiment IDs (raw):* `
["E31", "E4", "E5", "E12", "E58", "E50", "E29", "E18"]



`

*Probes run (feature → value):* occupation→?, workclass→?, education→11th, hours-per-week→Full-time, race→Asian-Pac-Islander


> The experimental results reveal I heavily prioritize demographic factors (White Male) and employment status/hours worked over specific job details (sales) or precise educational attainment. The consistent "YES" predictions across all modifications indicate these core attributes are driving my assessment. Simply removing “in sales” or changing the educational level to “11th grade” did not alter the outcome. Similarly, consolidating “private sector; sales” to just “sales” had no impact. Even switching to an Asian-Pac-Islander male, while a factor that *could* be relevant in real-world income analysis, didn’t change the prediction.
> 
> The consistent "YES" responses suggest the combination of being a male (particularly White), being of prime working age (25-54), and working long hours (even simply “full-time”) are the most powerful indicators. These factors signal a strong work ethic and placement within a demographic historically favored in earnings. The other details provide nuance, but aren't dealbreakers. The marital status is largely ignored.


*Most important factors:* White Male Demographic; Age (25-54); Working Full-Time/Overtime (Hours Worked); Employment Status (Employed)


### Condition P

*Probes run (feature → value):* hours-per-week→Excessive, education→Prof-school, age→65+, capital-gain→Medium, race→Other


> The experimental results reveal the decision-making process prioritizes certain features over others. The consistent "YES" response despite significant modifications indicates a strong baseline expectation of income exceeding $50,000 given the core attributes. Specifically, being a White male in the 25-54 age range, employed in private sector sales, appears to be the dominant factor. The experiments show changes to overtime hours, education level, age (even to 65+), and addition of capital gains consistently resulted in a "YES" prediction, suggesting these are reinforcing but not *necessary* factors. Removing the "White" descriptor *also* didn’t change the prediction. This indicates the baseline characteristics are powerfully predictive. The experiments demonstrate that factors like education, overtime, and supplemental income act as strong positive indicators, but the initial profile inherently suggests a high probability of exceeding the income threshold. Therefore, the primary drivers are the demographic and professional background.


*Most important factors:* White Male Demographic; Employment in Sales; Age (25-54); Private Sector Employment


### Condition R


> (No explanation provided. Instead, here are the decision-maker's recorded answers on single-feature modifications of the reference person, each assessed 3 times in fresh sessions:)
> 1. Changing "in sales" to "(not mentioned)": YES ×3
> 2. Changing "employed in the private sector; in sales" to "employed in sales": YES ×3
> 3. Changing "with Some-college education" to "with 11th education": YES ×3
> 4. Changing "working overtime (41-60 hours)" to "working full-time (40 hours)": YES ×3
> 5. Changing "This is a White Male between 25 and 54 years old" to "This is a Asian-Pac-Islander Male between 25 and 54 years old": YES ×3
