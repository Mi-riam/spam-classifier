# Spam Klasyfikator

Spam Klasyfikator to aplikacja służąca do klasyfikacji treści e-maili na podstawie ich zawartości, określając czy dany e-mail jest spamem, czy też nie. Projekt został stworzony przy użyciu języka Python oraz biblioteki Streamlit, co umożliwia prostą i intuicyjną prezentację wyników w przeglądarce.

## Wymagania
Aby uruchomić projekt, upewnij się, że masz zainstalowane:
- Python 3.7 lub nowszy
- Biblioteki wymagane do działania aplikacji (zdefiniowane w pliku `requirements.txt`)

## Instalacja
1. Zainstaluj wymagane biblioteki, wykonując poniższe polecenie w terminalu:
   ```bash
   pip install -r requirements.txt
   ```

## Uruchamianie aplikacji
Aby uruchomić aplikację Streamlit, wykonaj jedno z poniższych poleceń w terminalu:

```bash
streamlit run app.py
```

lub alternatywnie:

```bash
python -m streamlit run app.py
```

## Rozwiązywanie problemów
Jeśli podczas działania aplikacji napotkasz błędy związane z biblioteką `nltk`, wykonaj poniższe polecenia:

1. Zaktualizuj bibliotekę `nltk`:
   ```bash
   python -m pip install --upgrade nltk
   ```
2. Pobierz wymagane zasoby dla `nltk`:
   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

Po wykonaniu tych kroków, aplikacja powinna działać poprawnie.

## Funkcjonalności
- Analiza treści e-maila w czasie rzeczywistym
- Klasyfikacja wiadomości na kategorie: **Spam** lub **Nie-spam**
- Interfejs użytkownika w przeglądarce oparty na Streamlit

