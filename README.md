## id_strategic_ver, it_strategic_ver, idit_strategic_ver

- (도메인 지식) 특정 사업부(Business Unit이 ID일 때), 특정 사업 영역(Vertical Level1)에 대해 가중치를 부여
- (도메인 지식) 특정 사업부(Business Unit이 IT일 때), 특정 사업 영역(Vertical Level1)에 대해 가중치를 부여
- Id_strategic_ver이나 it_strategic_ver 값 중 하나라도 1의 값을 가지면 1 값으로 표현

### Vertical Level 1(사업 영역 - business_area): 

- corporate / office
- hotel & accommodation

## ver_cus

- 특정 Vertical Level 1(사업영역) 이면서 Customer_type(고객 유형)이 소비자(End-user)인 경우에 대한 가중치

### Vertical Level 1(사업 영역 - business_area):

- corporate / office
- hotel & accommodation
- education
- retail

## ver_pro

- 특정 Vertical Level 1(사업영역) 이면서 특정 Product Category(제품 유형)인 경우에 대한 가중치

### Vertical Level 1(사업 영역 - business_area) 별  product_category:

- corporate / office

    - standard signage
    - high brightness signage
    - interactive signage
    - video wall signage
    - led signage
    - signage care solution
    - oled signage
    - special signage
    - uhd signage
    - smart tv signage
    - signage care solutions
    - digital signage
    - monitor signage,commercial tv,monior/monitor tv
    - monitor signage,monior/monitor tv
    - monitor signage,commercial tv,monior/monitor tv,projector,tv
    - monitor signage,commercial tv,monior/monitor tv,tv
    - monitor signage,commercial tv,solar,ess,monior/monitor tv,pc,projector,robot,system ac,ems,rac,chill
    - tv signage
    - signage
    - monitor signage,tv

- hotel & accommodation

    - hotel tv

- retail

    - led signage
    - video wall signage
    - high brightness signage
    - standard signage
    - oled signage
    - interactive signage
    - special signage
    - smart tv signage
    - uhd signage
    - tv signage
    - signage care solution
    - ultra stretch signage
    - monitor signage,monior/monitor tv
    - monitor signage,commercial tv,solar,ess,monior/monitor tv,pc,projector,robot,system ac,ems,rac,chill
    - monitor signage,commercial tv,monior/monitor tv,pc,tv,home beauty,audio/video
    - monitor signage,monior/monitor tv,tv,audio/video
    - signage
    - digital signage
    - signage care solutions
    - monitor signage,monior/monitor tv,vacuum cleaner,tv,home beauty,commercial tv,pc,refrigerator,styler

#### corporate / office와 retail의 교집합

- digital signage
- high brightness signage
- interactive signage
- led signage
- monitor signage,commercial tv,solar,ess,monior/monitor tv,pc,projector,robot,system ac,ems,rac,chill
- monitor signage,monior/monitor tv
- oled signage
- signage
- signage care solution
- signage care solutions
- smart tv signage
- special signage
- standard signage
- tv signage
- uhd signage
- video wall signage

## ver_win_rate_x

- 전체 Lead 중에서 Vertical을 기준으로 Vertical 수 비율과 Vertical 별 Lead 수 대비 영업 전환 성공 비율 값을 곱한 값

### 각 ver_win_rate_x 별 business_area:

|ver_win_rate_x|business_area|trainset 개수|
|--------------|--------|-------------|
|0.0030792876608617|corporate / office|4097|
|0.0005719551277132|education|1727|
|0.0007167734380046|hotel & accommodation|2013|
|6.044033666058328e-05|hospital & health care|1199|
|0.0005432224318428|special purpose|1929|
|0.0002983104051378|residential (home)|1818|
|9.65915660650443e-05|government department|650|
|0.0011827288932506|retail|3011|
|0.0002153634176709|factory|1035|
|2.3159381337232847e-06|power plant / renewable energy|113|
|1.2765902883450302e-05|transportation|306|
|2.5889552307882245e-05|public facility|519|

## ver_win_ratio_per_bu

- 특정 Vertical Level1의 Business Unit 별 샘플 수 대비 영업 전환된 샘플 수의 비율을 계산

### 각 ver_win_ratio_per_bu 별 business_unit 및 business_area:

|ver_win_ratio_per_bu|business_unit|business_area|trainset 개수|
|--------------------|-------------|-------------|-------------|
|0.0268456375838926|AS      |corporate / office            |1278|
|0.0645661157024793|ID      |corporate / office            |2460|
|0.0344827586206896|Solution|corporate / office            |29  |
|0.048629531388152 |ID      |education                     |1356|
|0.0514705882352941|AS      |education                     |210 |
|0.0640703517587939|ID      |special purpose               |1075|
|0.022633744855967 |AS      |special purpose               |713 |
|0.1285714285714285|AS      |hospital & health care        |156 |
|0.1311475409836065|ID      |hospital & health care        |148 |
|0.0354838709677419|ID      |residential (home)            |385 |
|0.0201207243460764|AS      |residential (home)            |1365|
|0.0794117647058823|ID      |government department         |421 |
|0.0227272727272727|AS      |government department         |173 |
|0.0498402555910543|ID      |retail                        |2028|
|0.0115830115830115|AS      |retail                        |788 |
|0.071345029239766 |ID      |hotel & accommodation         |984 |
|0.0369127516778523|AS      |factory                       |403 |
|0.0609243697478991|ID      |factory                       |540 |
|0.2857142857142857|ID      |power plant / renewable energy|43  |
|0.2272727272727272|AS      |power plant / renewable energy|62  |
|0.0535714285714285|ID      |transportation                |216 |
|0.031578947368421 |ID      |public facility               |271 |
|0.0287769784172661|AS      |public facility               |200 |

### train_set: ver_win_ratio_per_bu가 nan일 때 business_area는 있는것들의 개수 = 3113
### train_set:business_area가 nan일 때 ver_win_ratio_per_bu는 있는것들의 개수 = 0
### train_set: 둘다 nan인 것들 개수 = 40882

### test_set: ver_win_ratio_per_bu가 nan일 때 business_area는 있는것들의 개수 = 467
### test_set:business_area가 nan일 때 ver_win_ratio_per_bu는 있는것들의 개수 = 0
### test_set: 둘다 nan인 것들 개수 = 2898

## com_reg_ver_win_rate

|com_reg_ver_win_rate|business_unit|business_area|customer_country|
|--------------------|-------------|-------------|----------------|
|0.0037878787878787:|['AS',|'residential (home)',|['argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'dominicanrepublic', 'elsalvador', 'honduras', 'mexico', 'nicaragua', 'panama', 'peru', 'stkitts', 'trinidadandtobago', 'unitedkingdom', 'uruguay']],|
|0.0039370078740157:|['AS',|'corporate / office',|['antigua', 'argentina', 'bahamas', 'belize', 'brazil', 'chile', 'colombia', 'dominicanrepublic', 'elsalvador', 'guatemala', 'honduras', 'jamaica', 'mexico', 'panama', 'peru', 'uruguay']],|
|0.0118577075098814:|['AS',|'special purpose',|['argentina', 'brazil', 'chile', 'colombia', 'dominicanrepublic', 'ecuador', 'elsalvador', 'guatemala', 'jamaica', 'mexico', 'panama', 'paraguay', 'peru', 'unitedkingdom', 'uruguay']],|
|0.0135135135135135:|['ID',|'special purpose',|['albania', 'belgium', 'bosniaandherzegovina', 'bulgaria', 'croatia', 'cyprus', 'czech', 'france', 'germany', 'greece', 'ireland', 'italy', 'latvia', 'netherlands', 'poland', 'portugal', 'romania', 'slovenia', 'spain', 'switzerland', 'unitedkingdom']],|
|0.0151515151515151:|['AS',|'education',|['argentina', 'brazil', 'chile', 'colombia', 'mexico', 'panama', 'peru']],|
|0.0175438596491228:|['ID',|'education',|['france', 'germany', 'greece', 'italy', 'kosovo', 'poland', 'portugal', 'spain', 'unitedkingdom']],|
|0.0181818181818181:|['AS',|'special purpose',|['australia', 'indonesia', 'japan', 'papuanewguinea', 'philippines', 'singapore', 'thailand', 'vietnam']],|
|0.0199004975124378:|['ID',|'corporate / office',|['albania', 'belgium', 'bulgaria', 'croatia', 'cyprus', 'czech', 'denmark', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy', 'luxembourg', 'malta', 'netherlands', 'poland', 'portugal', 'romania', 'serbia', 'slovenia', 'spain', 'sweden', 'switzerland', 'unitedkingdom']],|
|0.0311958405545927:|['ID',|'education',|['india', 'unitedstates']],|
|0.032258064516129:|['ID',|'education',|['afghanistan', 'bahrain', 'canada', 'egypt', 'ghana', 'israel', 'jordan', 'kenya', 'kuwait', 'nigeria', 'oman', 'saudiarabia', 'southafrica', 'türkiye', 'u.a.e', 'unitedstates', 'yemen']],|
|0.0327868852459016:|['ID',|'education',|['argentina', 'brazil', 'chile', 'colombia', 'ecuador', 'guatemala', 'jamaica', 'mexico', 'panama', 'peru', 'unitedkingdom']],|
|0.0330578512396694:|['ID',|'special purpose',|['india']],|
|0.0408163265306122:|['AS',|'corporate / office',|['afghanistan', 'democraticrepublicofthecongo', 'egypt', 'ethiopia', 'ghana', 'iraq', 'israel', 'jordan', 'kenya', 'mauritania', 'mozambique', 'nigeria', 'oman', 'pakistan', 'qatar', 'saudiarabia', 'southafrica', 'togo', 'türkiye', 'u.a.e', 'unitedrepublicoftanzania']],|
|0.043103448275862:|['ID',|'special purpose',|['afghanistan', 'angola', 'armenia', 'botswana', 'centralafricanrepublic', 'egypt', 'ethiopia', 'ghana', 'iraq', 'israel', 'kenya', 'kuwait', 'lebanon', 'libya', 'nigeria', 'oman', 'pakistan', 'palestine', 'qatar', 'saudiarabia', 'southafrica', 'türkiye', 'u.a.e', 'zambia']],|
|0.0434782608695652:|['IT',|'corporate / office',|['algeria', 'canada', 'egypt', 'kuwait', 'morocco', 'saudiarabia', 'türkiye', 'u.a.e', 'uganda', 'unitedstates']],|
|0.0446428571428571:|['ID',|'corporate / office',|['australia', 'brazil', 'canada', 'chile', 'china', 'clinton,ok73601', 'colombia', 'costarica', 'dominicanrepublic', 'ecuador', 'guatemala', 'honduras', 'mexico', 'panama', 'peru', 'saudiarabia', 'unitedstates']],|
|0.0485436893203883:|['AS',|'special purpose',|['egypt', 'iraq', 'jordan', 'kenya', 'lebanon', 'nigeria', 'oman', 'pakistan', 'saudiarabia', 'southafrica', 'sudan', 'swaziland', 'tunisia', 'türkiye', 'u.a.e', 'unitedrepublicoftanzania', 'zambia']],|
|0.0575342465753424:|['ID',|'corporate / office',|['india']],|
|0.0666666666666666:|['AS',|'corporate / office',|['australia', 'bangladesh', 'china', 'india', 'indonesia', 'papuanewguinea', 'philippines', 'singapore', 'srilanka', 'thailand', 'vietnam']],|
|0.0888888888888888:|['AS',|'corporate / office',|['india']],|
|0.004:|['AS',|'retail',|['antigua', 'argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'ecuador', 'elsalvador', 'guatemala', 'honduras', 'mexico', 'nicaragua', 'panama', 'paraguay', 'peru', 'puertorico', 'trinidadandtobago', 'unitedkingdom', 'venezuela']],|
|0.0109890109890109:|['AS',|'residential (home)',|['australia', 'bangladesh', 'indonesia', 'myanmar', 'newzealand', 'papuanewguinea', 'philippines', 'singapore', 'thailand', 'vietnam']],|
|0.0169491525423728:|['ID',|'hotel & accommodation',|['albania', 'belgium', 'czech', 'denmark', 'france', 'germany', 'greece', 'hungary', 'italy', 'malta', 'netherlands', 'norway', 'poland', 'portugal', 'romania', 'serbia', 'spain', 'sweden', 'switzerland', 'unitedkingdom']],|
|0.0172413793103448:|['ID',|'residential (home)',|['india']],|
|0.0196078431372549:|['AS',|'retail',|['algeria', 'egypt', 'iraq', 'kenya', 'kuwait', 'mauritius', 'nigeria', 'oman', 'pakistan', 'palestine', 'qatar', 'saudiarabia', 'serbia', 'southafrica', 'türkiye', 'u.a.e', 'yemen']],|
|0.0202020202020202:|['AS',|'factory',|['algeria', 'burkinafaso', 'egypt', 'iran', 'iraq', 'kenya', 'morocco', 'nigeria', 'oman', 'pakistan', 'saudiarabia', 'senegal', 'southafrica', 'türkiye', 'u.a.e']],|
|0.0227272727272727:|['AS',|'retail',|['australia', 'bangladesh', 'indonesia', 'papuanewguinea', 'philippines', 'singapore', 'thailand', 'vietnam']],|
|0.025:|['ID',|'transportation',|['belgium', 'bulgaria', 'france', 'germany', 'hungary', 'italy', 'poland', 'portugal', 'slovenia', 'spain', 'sweden', 'switzerland', 'unitedkingdom']],|
|0.0289256198347107:|['ID',|'retail',|['india']],|
|0.0289855072463768:|['AS',|'retail',|['india']],|
|0.036036036036036:|['ID',|'factory',|['india']],|
|0.037037037037037:|['AS',|'residential (home)',|['azerbaijan', 'belgium', 'bosniaandherzegovina', 'bulgaria', 'egypt', 'ethiopia', 'france', 'germany', 'greece', 'hungary', 'israel', 'oman', 'poland', 'portugal', 'saudiarabia', 'türkiye', 'u.a.e', 'unitedkingdom', 'unitedrepublicoftanzania', 'zimbabwe']],|
|0.04:|['AS',|'government department',|['afghanistan', 'algeria', 'australia', 'bangladesh', 'brazil', 'chile', 'colombia', 'egypt', 'ghana', 'honduras', 'indonesia', 'iraq', 'israel', 'kenya', 'kuwait', 'libya', 'mauritius', 'mexico', 'morocco', 'nigeria', 'oman', 'pakistan', 'peru', 'philippines', 'qatar', 'saudiarabia', 'singapore', 'southafrica', 'srilanka', 'thailand', 'türkiye', 'u.a.e', 'uganda', 'yemen']],|
|0.0416666666666666:|['AS',|'government department',|['albania', 'bulgaria', 'egypt', 'france', 'germany', 'greece', 'hongkong', 'hungary', 'iraq', 'italy', 'jordan', 'kuwait', 'netherlands', 'nigeria', 'poland', 'portugal', 'romania', 'saudiarabia', 'spain', 'sweden', 'switzerland', 'u.a.e', 'uganda', 'unitedkingdom']],|
|0.0422535211267605:|['ID',|'public facility',|['argentina', 'brazil', 'chile', 'colombia', 'guatemala', 'mexico', 'peru', 'puertorico']],|
|0.0476190476190476:|['ID',|'transportation',|['argentina', 'bahamas', 'brazil', 'chile', 'colombia', 'costarica', 'dominicanrepublic', 'ecuador', 'guatemala', 'mexico', 'panama', 'peru', 'puertorico', 'stmaarten', 'unitedkingdom']],|
|0.0491803278688524:|['ID',|'retail',|['afghanistan', 'azerbaijan', 'egypt', 'ethiopia', 'georgia', 'ghana', 'israel', 'kenya', 'kuwait', 'libya', 'morocco', 'mozambique', 'namibia', 'nigeria', 'oman', 'pakistan', 'qatar', 'rwanda', 'saudiarabia', 'southafrica', 'sudan', 'tunisia', 'türkiye', 'u.a.e', 'uganda', 'unitedrepublicoftanzania', 'zimbabwe']],|
|0.0496894409937888:|['AS',|'residential (home)',|['bahrain', 'burkinafaso', 'egypt', 'gabon', 'ghana', 'guinea', 'iran', 'iraq', 'israel', 'jordan', 'kuwait', 'lebanon', 'nigeria', 'saudiarabia', 'southafrica', 'syria', 'türkiye', 'u.a.e', 'unitedrepublicoftanzania']],|
|0.0531914893617021:|['ID',|'government department',|['india']],|
|0.0538922155688622:|['ID',|'residential (home)',|['argentina', 'brazil', 'chile', 'colombia', 'costarica', 'ecuador', 'elsalvador', 'mexico', 'panama', 'peru', 'trinidadandtobago', 'unitedkingdom']],|
|0.0544217687074829:|['ID',|'government department',|['argentina', 'bahamas', 'brazil', 'chile', 'colombia', 'mexico', 'nicaragua', 'panama', 'peru', 'unitedkingdom']],|
|0.0555555555555555:|['AS',|'education',|['bahrain', 'egypt', 'ghana', 'jordan', 'nigeria', 'saudiarabia', 'türkiye', 'u.a.e']],|
|0.0677966101694915:|['AS',|'residential (home)',|['india']],|
|0.0681818181818181:|['AS',|'hospital & health care',|['brazil', 'chile', 'colombia', 'ecuador', 'mexico', 'panama', 'peru', 'venezuela']],|
|0.0695652173913043:|['ID',|'factory',|['argentina', 'brazil', 'chile', 'colombia', 'dominicanrepublic', 'elsalvador', 'mexico', 'panama', 'peru', 'unitedkingdom']],|
|0.0714285714285714:|['AS',|'government department',|['india']],|
|0.0732484076433121:|['ID',|'retail',|['argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'dominicanrepublic', 'ecuador', 'guatemala', 'honduras', 'jamaica', 'mexico', 'panama', 'peru', 'puertorico', 'trinidadandtobago', 'unitedkingdom', 'venezuela']],|
|0.0749486652977412:|['ID',|'corporate / office',|['argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'dominicanrepublic', 'ecuador', 'elsalvador', 'guatemala', 'honduras', 'jamaica', 'mexico', 'nicaragua', 'panama', 'paraguay', 'peru', 'puertorico', 'unitedkingdom', 'venezuela']],|
|0.075:|['ID',|'corporate / office',|['afghanistan', 'algeria', "coted'ivoire", 'egypt', 'ethiopia', 'ghana', 'iran', 'israel', 'kenya', 'kuwait', 'morocco', 'namibia', 'nigeria', 'oman', 'pakistan', 'qatar', 'saudiarabia', 'senegal', 'southafrica', 'tunisia', 'türkiye', 'u.a.e', 'uganda', 'unitedrepublicoftanzania', 'yemen']],|
|0.0806916426512968:|['ID',|'special purpose',|['argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'guatemala', 'jamaica', 'mexico', 'panama', 'peru', 'puertorico', 'unitedkingdom']],|
|0.0833333333333333:|['ID',|'hospital & health care',|['india']],|
|0.0843373493975903:|['ID',|'corporate / office',|['australia', 'china', 'fiji', 'hongkong', 'indonesia', 'japan', 'malaysia', 'maldives', 'philippines', 'singapore', 'srilanka', 'taiwan', 'vietnam']],|
|0.0869565217391304:|['ID',|'government department',|['democraticrepublicofthecongo', 'egypt', 'ethiopia', 'israel', 'kuwait', 'nigeria', 'qatar', 'saudiarabia', 'southafrica', 'türkiye', 'u.a.e', 'zambia']],|
|0.1052631578947368:|['ID',|'government department',|['bulgaria', 'czech', 'germany', 'ireland', 'italy', 'malta', 'poland', 'portugal', 'spain', 'unitedkingdom']],|
|0.1136363636363636:|['ID',|'factory',|['azerbaijan', 'egypt', 'iran', 'israel', 'kenya', 'nigeria', 'saudiarabia', 'southafrica', 'türkiye', 'u.a.e', 'yemen']],|
|0.1162790697674418:|['ID',|'special purpose',|['australia', 'china', 'hongkong', 'indonesia', 'japan', 'laos', 'malaysia', 'newzealand', 'papuanewguinea', 'philippines', 'singapore', 'srilanka', 'taiwan', 'thailand']],|
|0.1184210526315789:|['ID',|'retail',|['australia', 'brunei', 'cambodia', 'china', 'hongkong', 'indonesia', 'japan', 'malaysia', 'maldives', 'nepal', 'philippines', 'singapore', 'taiwan', 'thailand', 'unitedstates', 'vietnam']],|
|0.1186440677966101:|['ID',|'retail',|['australia', 'canada', 'hongkong', 'unitedstates']],|
|0.1241217798594847:|['ID',|'hotel & accommodation',|['argentina', 'bahamas', 'brazil', 'chile', 'colombia', 'ecuador', 'guatemala', 'honduras', 'mexico', 'panama', 'peru', 'puertorico', 'saintlucia', 'unitedkingdom', 'unitedstates']],|
|0.125:|['IT',|'retail',|['egypt', 'saudiarabia', 'southafrica', 'u.a.e']],|
|0.1363636363636363:|['AS',|'factory',|['australia', 'china', 'indonesia', 'philippines', 'singapore', 'vietnam']],|
|0.1470588235294117:|['AS',|'education',|['india']],|
|0.1666666666666666:|['ID',|'residential (home)',|['canada', 'unitedstates']],|
|0.1818181818181818:|['IT',|'hospital & health care',|['china', 'egypt', 'hongkong', 'indonesia', 'japan', 'malaysia', 'philippines', 'qatar', 'singapore', 'southafrica', 'taiwan', 'türkiye', 'u.a.e', 'vietnam', 'yemen']],|
|0.2:|['IT',|'public facility',|['costarica', 'dominicanrepublic', 'guatemala']],|
|0.2142857142857142:|['ID',|'education',|['australia', 'bangladesh', 'brunei', 'hongkong', 'indonesia', 'japan', 'myanmar', 'philippines', 'singapore', 'taiwan', 'thailand', 'vietnam']],|
|0.2307692307692307:|['IT',|'hospital & health care',|['austria', 'bulgaria', 'cyprus', 'france', 'germany', 'italy', 'poland', 'spain', 'switzerland', 'unitedkingdom']],|
|0.25:|['ID',|'power plant / renewable energy',|['argentina', 'brazil', 'chile', 'mexico', 'peru']],|
|0.2692307692307692:|['ID',|'hospital & health care',|['argentina', 'brazil', 'chile', 'colombia', 'costarica', 'mexico', 'peru']],|
|0.3333333333333333:|['ID',|'special purpose',|['brazil', 'chile', 'colombia', 'germany', 'india', 'mexico', 'peru', 'unitedstates']],|
|0.3636363636363636:|['IT',|'residential (home)',|['colombia', 'costarica', 'ecuador', 'mexico', 'peru']],|
|0.3902439024390244:|['ID',|'education',|['canada', 'unitedstates']],|
|0.4:|['ID',|'public facility',|['canada', 'unitedstates']],|
|0.4444444444444444:|['AS',|'power plant / renewable energy',|['mozambique', 'nigeria', 'qatar', 'senegal', 'türkiye', 'u.a.e']],|
|0.4615384615384615:|['AS',|'hospital & health care',|['india']],|
|0.5:|['IT',|'factory',|['canada', 'unitedstates']],|
|0.6153846153846154:|['ID',|'government department',|['canada', 'israel', 'unitedstates']],|
|0.6428571428571429:|['IT',|'hospital & health care',|['canada', 'unitedstates']],|
|0.8333333333333334:|['ID',|'transportation',|['canada', 'unitedstates']],|
|1.0:|['ID',|'power plant / renewable energy',|['australia', 'brazil', 'indonesia', 'malaysia', 'philippines', 'vietnam']],|