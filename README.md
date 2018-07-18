# Belangrijke resultaten

## Bounds en dichtheid oplossingen

![alt text](Figures/distribution_normal.png)

In bovenstaand figuur is de scoredistributie van willekeurige, geldige oplossingen
getoond voor wijk 1 met de initiële plaatsing van de batterijen. Verder geeft
stippellijn de lower bound aan. Deze is berekend door alle Manhattan-afstanden
van de huizen tot hun dichtstbijzijnde batterij bij elkaar op te tellen.

Verder is aan het aantal ongeldige oplossingen te zien de oplossingsdichtheid
zeer laag is. Slechts 1% van de willekeurige oplossingen lijkt geldig te zijn. 

## Batterijen verplaatsen

![alt text](Figures/reposition_hillclimber.png)

Bovenstaand figuur toont de posities van de huizen en batterijen in wijk 1 (bovenste
rij figuren). Van de output van de huizen en de capaciteit van de batterijen is
met behulp van gaussische verdelingen (standaarddeviatie = 10) een heat map gemaakt
(middelste rij figuren). Dit geeft een goede indicatie van waar de batterijen beter
geplaatst kunnen worden. Als de output van de huizen en de capaciteit over elkaar
heen legt (onderste rij figuren) wil je dat map zo wit mogelijk is (d.w.z. output
en capaciteit in balans). Na hill climben op deze heat map score komt er een betere
positionering van de batterijen uit...

![alt text](Figures/distribution_replace.png)

In bovenstaand figuur is te zien hoe de scoredistributie van de willekeurige
oplossingen én lower bound is verschoven. Dit duidt erop dat met deze nieuwe
positionering van de batterijen lagere kosten gerealiseerd kunnen worden.

## Vergelijking scores na verplaatsen batterijen

![alt text](Figures/results_table_1.png)

Deze tabel toont de hoogst behaalde scores van de verschillende algoritmes. Hiervoor
zijn de batterijposities gebruikt die in de vorige paragraaf worden besproken.
Voor de hill climber en de simulated annealing is het resultaat van de branch 'n bound
als startpunt gebruikt. Bovendien is er een simulated annealing gedaan vanuit het
resultaat van hill climber. Het cooling scheme van simulated annealing is exponentieel.

## Het plaatsen van verschillende typen batterijen

![alt text](Figures/Sigma_experiments/heat_simannealing_sigma10_2.png)

Het linkerfiguur van bovenstaande figuren toont dat een goede score van de
heat map een goede indicatie is van de score die met simulated annealing voor een
bepaalde batterijpositionering behaald kan worden. Ook zie je dat de lower bound
net zo een goede indicator lijkt hiervoor. Echter, in deze figuren hebben de batterijen
een gelijke capaciteit. Dat is een vereiste voor het gebruik van de lower bound
als indicator. Als er verschillende typen batterijen geplaatst moeten worden, dan
moet je de heat map score gebruiken, omdat die dat verschil in typen meeneemt.
Een toekomstig figuur zal dit aantonen.

![alt text](Figures/n_batteries_vs_cable_cost.png)

Bovenstaand figuur lijkt een redelijk logisch beeld te tonen. Hoe meer batterijen
hoe lager de kabelkosten. Volgend figuur lijkt echter ook iets belangrijks te tonen...

![alt text](Figures/n_batteries_vs_lower_bound_2.png)

Bovenstaand figuur toont aan dat de korting op de kabelkosten teniet wordt gedaan
als je ook de kosten van de batterijen meeneemt. Hierdoor lijkt het erop dat, zolang
je precies genoeg capaciteit in het grid hebt voor de output van de huizen, het niet
zoveel uit maakt hoeveel batterijen er geplaatst worden. Dit figuur heeft echter
wel de lower bound op de y-as staan en niet een score van, bijvoorbeeld, simulated
annealing.

## Nog missende figuren

* De eindscores voor de beste (door ons gevonden) positionering van de nieuwe
batterijen moeten getoond worden.

* We hebben voor simulated annealing 3 cooling schemes met elkaar vergeleken:
lineair, exponentieel en sigmoïdaal. Het lijkt erop dat sigmoïdaal het beste werkt,
daarna exponentieel en lineair het minst goed. Deze vergelijking moet nog in cijfers
getoond worden.
