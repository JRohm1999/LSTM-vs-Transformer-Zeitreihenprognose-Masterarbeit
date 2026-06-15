# Vergleich von LSTM- und Transformer-Modellen zur Zeitreihenprognose im Unternehmenskontext

Dieses Repository enthält den gesamten Code der für das Preprocessing bis hin zum Evaluieren der Modelle benötigt wird und in der Masterarbeit verwendet wurde. 
Ebenso ist eine Notebook enthalten, welches für das Cloud-Training erstellt wurde. Dieses Notebook importiert das Repository, Importiert alle Biblitheken aus der requirements.py Datei und ruft die Traningsskripte auf. Anpassungen wie Optunaläufe oder Parametereinstellungen erfolgen in den Skripten selbst.

Dieses Repository besteht aus:

- Code
- Ordnerstruktur für den Import des Subsets
- Trainierten Modellen
- Ergebnissen: Traningsläufe & finalen Testruns

Der Datensatz und das finale Subset sind aufgrund der Größe nicht im Repository enthalten und müssen für einen Test des Codes selbständig aus den Rohdaten des M5-Datensatzes erstellt werden. Dafür müssen die Daten von dem M5 Kaggel Wettbewerb heruntergeladen werden und auf Basis dieser Daten das preprocessing.py und das create_subset.py Skript ausgeführt werden.
Anschließend sollte der Subset Datensatz für das Traning oder Testen der Modelle in die Ordnerstruktur data/preprocessed überführt werden, damit die Tranings- und Testskripte auf die Daten zugreifen können.



