# 📋 Requirements Sheet – Strava Supercompensation Tool (Python)

## 1. Ziel
- Automatisiertes Tool, das Strava-Aktivitäten über die **Strava API** abruft.  
- Analyse der Daten anhand eines **Superkompensations-/Banister-Modells**.  
- Ausgabe einer **Trainingsempfehlung** („Pause“, „Locker“, „Hart“).  

---

## 2. Kernfunktionen (MVP)
1. **Strava API-Anbindung**
   - OAuth2 Login/Token Refresh  
   - Abruf von Aktivitäten (Datum, Dauer, Distanz, Durchschnitts-HF, Relative Effort, TSS falls verfügbar)

2. **Datenspeicherung**
   - Lokale SQLite/PostgreSQL DB (je nach Anspruch)  
   - Tabellen: `activities`, `metrics`, `model_state`

3. **Superkompensationsmodell**
   - Implementierung eines einfachen Impulse-Response Modells (Banister)  
   - Parameter: Fitness, Fatigue, Form  
   - Anpassbare Konstanten (k1, k2, decay rates)

4. **Auswertung & Empfehlung**
   - Täglicher Status:  
     - Fatigue hoch → Pause  
     - Superkompensation → Harte Einheit  
     - Neutral → Locker  
   - Ausgabe: CLI / einfache HTML Seite / JSON API

---

## 3. Erweiterungen (Phase 2)
- Einbindung von **HRV & Schlafdaten** (z. B. Garmin, Oura API).  
- Visualisierung (Matplotlib/Plotly Dash).  
- Automatische Push-Benachrichtigung (Telegram Bot, E-Mail).  
- Anpassbares Modell pro Nutzer (Machine Learning).  

---

## 4. Technische Anforderungen
- **Programmiersprache**: Python 3.10+  
- **Libraries (Basis)**:
  - `requests` (API Calls)  
  - `pandas` (Datenhandling)  
  - `sqlalchemy` oder `sqlite3` (Datenbank)  
  - `matplotlib` oder `plotly` (Visualisierung, optional)  
  - `fastapi` (falls Web-API gewünscht)  

- **Deployment**
  - Lokal ausführbar (Mac/Linux)  
  - Optional Docker-Container für Serverbetrieb  

---

## 5. Non-Functional Requirements
- **Erweiterbar**: Modellparameter und Datenquellen leicht anpassbar  
- **Reproduzierbar**: Daten + Berechnungen nachvollziehbar speichern  
- **Datensicherheit**: Strava OAuth Tokens sicher speichern (z. B. `.env` oder Secret Store)  
- **Performance**: Abruf und Berechnung < 10 Sekunden pro Lauf  

---

## 6. Deliverables
- Python Package/Projektstruktur  
- Datenbank mit Aktivitäts-Historie  
- Skript zum Abrufen & Aktualisieren  
- Analyse-Modul (Superkompensation)  
- Erste CLI-Ausgabe (Textempfehlung)  
