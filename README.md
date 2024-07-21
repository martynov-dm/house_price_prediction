# ml_service

Пока что в сервисе есть 2 блока - авторизация/регистрация и генерация картинки по тексту.
Используется `huggingface inference api`.

## Запуск сервиса

Но нужны .env, которые не понятно откуда вам брать
`git clone`
`poetry install`
`docker-compose up --build --force-recreate --no-deps` or `poe up`
