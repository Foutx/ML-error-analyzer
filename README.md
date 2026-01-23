# API KEYS get here -> [Link](https://console.groq.com) 

# ML Error Analyzer

## 🐳 Запуск через Docker

### Собрать локально
```bash
git clone https://github.com/Foutx/ML-error-analyzer.git
cd ML-error-analyzer
docker build -t ml-analyzer -f ml_error_analyzer.dockerfile .
docker run ml-analyzer
```