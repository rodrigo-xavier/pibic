Primeiro instale o pyenv, para obter um ambiente python limpo.

O código a seguir executa o instalador automático do pyenv.

    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

Em seguida, instale o python versão 3.8.3 utilizando o pacote pyenv, da seguinte forma:

    pyenv install 3.5.9
    pyenv shell 3.5.9

Faça um clone do meu repositório:

    git clone https://github.com/rodrigo-xavier/about-me.git

Agora crie uma virtualenv, ative-a, faça o upgrade do pip e instale os requisitos do projeto:

    python -m venv virtualenv
    source virtualenv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
