# Web Scraping utilizando biblioteca BeautifulSoup e Selenium
# BeautifulSoup = analisar e extrair informações de arquivos HTML

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import pandas as pd

# database: Artificial Analysis - API providers

# Parallel Queries: Number of requests to the API simultaneously
    # Multiple is representaive of 10 simultaneous requests

# Prompt Length: Length of prompt (information provided and questions) to the model.
    # Short prompt represents ~80 input tokens, Moderate prompts are ~1,000 tokens and Long prompt ~10,000 input tokens.
    # Longer prompts are highly relevant to RAG (Retrieval-Augmented Generation) use-cases.
    # Coding prompts are ~1,000 tokens in length and focus on code generation tasks across a popular programming languages.
    # We separate coding prompts from the others as coding is low-entropy and differences in performance can be experienced.

#____________________________________________________________________

## CENÁRIOS ##
## Artificial Analysis - API providers ##

# CENÁRIO 1
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 100 tokens (short)

# CENÁRIO 2
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 1k tokens (medium)
        
# CENÁRIO 3
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 10k tokens (long)
        
# CENÁRIO 4
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 100k tokens
        
# CENÁRIO 5
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: Coding (1k tokens - medium)
        
# CENÁRIO 6
    # PROMPT OPTIONS:
        # Parallel Queries: Multiple
        # Prompt Length: 1k tokens (medium)
        
#______________________________________________________________________
#%% API Providers

# Categorias desejadas
parallel_queries = ["single", "multiple"]
prompt_length = ["short", "medium", "long", "100k", "medium_coding"]

# Lista para armazenar os dados raspados
rows = []

for queries in parallel_queries:
    if queries == "single":
        for length in prompt_length:
            driver = webdriver.Edge()

            # download the target page
            url = f"https://artificialanalysis.ai/leaderboards/providers/prompt-options/{queries}/{length}/"
            driver.get(url)

            # Esperar a página carregar e o botão aparecer
            wait = WebDriverWait(driver, 15)
            expand_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Expand")]')))
            expand_button.click()

            # Esperar a tabela expandida ser carregada
            time.sleep(3)

            # Localiza a tabela com classe "w-full"
            table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.w-full")))

            # Pega o HTML da tabela inteira
            html_table = table_element.get_attribute("outerHTML")

            # Fechar o navegador
            driver.quit()

            # Usar BeautifulSoup para parsear o HTML e converter para DataFrame
            soup = BeautifulSoup(html_table, "html.parser")

            # Extrair as linhas
            for row in soup.find_all("tr")[2:]:  # pula cabeçalhos
                cols = []
                tds = row.find_all("td")
                for i, td in enumerate(tds):
                    if i == 0:  # primeira coluna: pegar alt da imagem
                        img = td.find("img")
                        if img and img.has_attr("alt"):
                            cols.append(img["alt"])
                        else:
                            cols.append(td.text.strip())
                    else:
                        cols.append(td.text.strip())
                cols.append(queries)
                cols.append(length)
                if cols:
                    rows.append(cols)
    else:
        length = "medium"
        driver = webdriver.Edge()

        # download the target page
        url = f"https://artificialanalysis.ai/leaderboards/providers/prompt-options/{queries}/{length}/"
        driver.get(url)

        # Esperar a página carregar e o botão aparecer
        wait = WebDriverWait(driver, 15)
        expand_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Expand")]')))
        expand_button.click()

        # Esperar a tabela expandida ser carregada (ajuste o tempo se necessário)
        time.sleep(3)

        # Localiza a tabela com classe "w-full"
        table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.w-full")))

        # Pega o HTML da tabela inteira
        html_table = table_element.get_attribute("outerHTML")

        # Fechar o navegador
        driver.quit()

        # Usar BeautifulSoup para parsear o HTML e converter para DataFrame
        soup = BeautifulSoup(html_table, "html.parser")

        # Extrair as linhas
        for row in soup.find_all("tr")[2:]:  # pula cabeçalhos
            cols = []
            tds = row.find_all("td")
            for i, td in enumerate(tds):
                if i == 0:  # primeira coluna: pegar alt da imagem
                    img = td.find("img")
                    if img and img.has_attr("alt"):
                        cols.append(img["alt"])
                    else:
                        cols.append(td.text.strip())
                else:
                    cols.append(td.text.strip())
            cols.append(queries)
            cols.append(length)
            if cols:
                rows.append(cols)

# Extrair os headers
# Encontra todos os <tr> da tabela
all_trs = soup.find_all("tr")

# Assume que o segundo <tr> contém os headers (index 1)
header_row = all_trs[1]
headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
headers.append("Parallel Queries")
headers.append("Prompt Length")

# Criar DataFrame
df = pd.DataFrame(rows, columns=headers)

# Salvar como CSV
df.to_csv("BD-LLMs-AA-all-v2.csv", index=False, encoding='utf-8-sig')

print("Dados salvos como BD-LLMs-AA-all.csv")


#%% Somente modelos

# Categorias desejadas
parallel_queries = ["single", "multiple"]
prompt_length = ["short", "medium", "long", "100k", "medium_coding"]

# Lista para armazenar os dados raspados
rows = []

for queries in parallel_queries:
    if queries == "single":
        for length in prompt_length:
            driver = webdriver.Edge()

            # download the target page
            url = f"https://artificialanalysis.ai/leaderboards/models/prompt-options/{queries}/{length}/"
            driver.get(url)

            # Esperar a página carregar e o botão aparecer
            wait = WebDriverWait(driver, 15)
            botao = wait.until(EC.presence_of_element_located((By.XPATH, '//button[contains(text(), "Expand")]')))  # troque pelo seu seletor

            # Scrolla até o botão (centralizando na tela)
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", botao)
            time.sleep(0.5)  # Espera o scroll terminar
            
            expand_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Expand")]')))
            expand_button.click()

            # Esperar a tabela expandida ser carregada (ajuste o tempo se necessário)
            time.sleep(3)

            # Localiza a tabela com classe "w-full"
            table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.w-full")))

            # Pega o HTML da tabela inteira
            html_table = table_element.get_attribute("outerHTML")

            # Fechar o navegador
            driver.quit()

            # Usar BeautifulSoup para parsear o HTML e converter para DataFrame
            soup = BeautifulSoup(html_table, "html.parser")

            # Extrair as linhas
            for row in soup.find_all("tr")[2:]:  # pula cabeçalhos
                cols = []
                tds = row.find_all("td")
                for i, td in enumerate(tds):
                    if i == 1:  # primeira coluna: pegar alt da imagem
                        img = td.find("img")
                        if img and img.has_attr("alt"):
                            cols.append(img["alt"])
                        else:
                            cols.append(td.text.strip())
                    else:
                        cols.append(td.text.strip())
                cols.append(queries)
                cols.append(length)
                if cols:
                    rows.append(cols)
    else:
        length = "medium"
        driver = webdriver.Edge()

        # download the target page
        url = f"https://artificialanalysis.ai/leaderboards/models/prompt-options/{queries}/{length}/"
        driver.get(url)

        # Esperar a página carregar e o botão aparecer
        wait = WebDriverWait(driver, 15)
        botao = wait.until(EC.presence_of_element_located((By.XPATH, '//button[contains(text(), "Expand")]')))  # troque pelo seu seletor

        # Scrolla até o botão (centralizando na tela)
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", botao)
        time.sleep(0.5)  # Espera o scroll terminar
       
        expand_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Expand")]')))
        expand_button.click()

        # Esperar a tabela expandida ser carregada (ajuste o tempo se necessário)
        time.sleep(3)

        # Localiza a tabela com classe "w-full"
        table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.w-full")))

        # Pega o HTML da tabela inteira
        html_table = table_element.get_attribute("outerHTML")

        # Fechar o navegador
        driver.quit()

        # Usar BeautifulSoup para parsear o HTML e converter para DataFrame
        soup = BeautifulSoup(html_table, "html.parser")

        # Extrair as linhas
        for row in soup.find_all("tr")[2:]:  # pula cabeçalhos
            cols = []
            tds = row.find_all("td")
            for i, td in enumerate(tds):
                if i == 1:  # primeira coluna: pegar alt da imagem
                    img = td.find("img")
                    if img and img.has_attr("alt"):
                        cols.append(img["alt"])
                    else:
                        cols.append(td.text.strip())
                else:
                    cols.append(td.text.strip())
            cols.append(queries)
            cols.append(length)
            if cols:
                rows.append(cols)

# Extrair os headers
# Encontra todos os <tr> da tabela
all_trs = soup.find_all("tr")

# Assume que o segundo <tr> contém os headers (index 1)
header_row = all_trs[1]
headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
headers.append("Parallel Queries")
headers.append("Prompt Length")

# Criar DataFrame
df = pd.DataFrame(rows, columns=headers)

# Salvar como CSV
df.to_csv("BD-LLMs-AA-all-v2_models.csv", index=False, encoding='utf-8-sig')

print("Dados salvos como BD-LLMs-AA-all_models.csv")