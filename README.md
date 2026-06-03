# BwRC_projekt3

Celem trzeciego projektu realizowanego w ramach zajęć laboratoryjnych jest opracowanie
multimodalnego systemu biometrycznego, który pozwoli na weryfikację tożsamości
użytkownika.
• Należy skorzystać z dwóch cech mierzalnych – czyli obrazu twarzy, a także z szybkości
pisania na klawiaturze.
• W tym projekcie proszę skorzystać z modułu szybkości pisania na klawiaturze, który
przygotowaliście w ramach projektu numer #2 (proszę nie pisać go od nowa!).

System powinien działać w sposób następujący:
• W pierwszym kroku użytkownik podaje kim jest (login, imię lub jakikolwiek inny dodatkowy wyróżnik)
• Następnie system pobiera (przy użyciu kamery laptopa) obraz jego twarzy
• W trzecim kroku system prosi o wpisanie tekstu na klawiaturze (rodzaj tekstu, jego treść, a także sposób doboru treści pozostawiam do Państwa decyzji).
• Następnie system dokonuje kombinacji informacji pobranych od użytkownika i wydaje werdykt o
rozpoznaniu (bądź nie) użytkownika.

W przypadku kombinacji cech mogą Państwo podejść do niej w sposób następujący:
• Do oceny tożsamości z wykorzystaniem obrazu twarzy, mogą Państwo wykorzystać algorytm
Eigenfaces, który znajduje się w pakiecie OpenCV (algorytm z roku 1991, trochę nieprzystający do dzisiejszych wymagań, ale… warto się z nim zapoznać)
• Do oceny tożsamości przy użyciu szybkości pisania na klawiaturze, proszę skorzystać z modułu wypracowanego w projekcie numer #2.
• Kombinacja cech może odbywać się, na przykład, jako pełna zgodność obu modułów, decyzja ważona lub przy wykorzystaniu dodatkowych reguł.