#!/bin/sh

PROJECT_DIR=$(pwd)

FRONTEND_DIR=src/interfaces/coral_web
FRONTEND_BUILD_DIR=deploy/frontend
rm -rf $FRONTEND_BUILD_DIR
mkdir -p $FRONTEND_BUILD_DIR
cp -r $FRONTEND_DIR/public $FRONTEND_BUILD_DIR/
cp -r $FRONTEND_DIR/src $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/package.json $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/yarn.lock* $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/package-lock.json* $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/pnpm-lock.yaml* $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/next.config.mjs $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/tsconfig.json $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/tailwind.config.js $FRONTEND_BUILD_DIR/
cp $FRONTEND_DIR/postcss.config.js $FRONTEND_BUILD_DIR/
source .deployenv
echo "NEXT_PUBLIC_API_HOSTNAME=$NEXT_PUBLIC_API_HOSTNAME" > $FRONTEND_BUILD_DIR/.env.development
echo "NEXT_PUBLIC_HAS_CUSTOM_LOGO=true" >> $FRONTEND_BUILD_DIR/.env.development
echo "NEXT_PUBLIC_API_HOSTNAME=$NEXT_PUBLIC_API_HOSTNAME" > $FRONTEND_BUILD_DIR/.env.production
echo "NEXT_PUBLIC_HAS_CUSTOM_LOGO=true" >> $FRONTEND_BUILD_DIR/.env.production
rm -f $PROJECT_DIR/frontend-deploy.zip
cd $FRONTEND_BUILD_DIR
zip -vr $PROJECT_DIR/frontend-deploy.zip ./* ./.*
cd $PROJECT_DIR
az webapp deploy --async --type zip --resource-group openai-chat-app --name cohere-chat-app --src-path $PROJECT_DIR/frontend-deploy.zip


BACKEND_DIR=src
BACKEND_BUILD_DIR=deploy/backend
rm -rf $BACKEND_BUILD_DIR
mkdir -p $BACKEND_BUILD_DIR
cp pyproject.toml $BACKEND_BUILD_DIR/
cp poetry.lock $BACKEND_BUILD_DIR/
cp -r $BACKEND_DIR/backend $BACKEND_BUILD_DIR/
cp -r $BACKEND_DIR/community $BACKEND_BUILD_DIR/
rm -rf $BACKEND_BUILD_DIR/backend/data
mkdir -p $BACKEND_BUILD_DIR/backend/data
source .deployenv
echo "DATABASE_URL=$DATABASE_URL" > $BACKEND_BUILD_DIR/.env
echo "USE_COMMUNITY_FEATURES=$USE_COMMUNITY_FEATURES" >> $BACKEND_BUILD_DIR/.env
echo "COHERE_API_KEY=$COHERE_API_KEY" >> $BACKEND_BUILD_DIR/.env
docker compose run --rm --build backend bash -c "pip freeze >> /workspace/tmp/requirement.txt" 
rm -f $PROJECT_DIR/backend-deploy.zip
cd $BACKEND_BUILD_DIR
zip -vr $PROJECT_DIR/backend-deploy.zip ./* ./.*
cd $PROJECT_DIR
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings PYTHONDONTWRITEBYTECODE=1
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings PYTHONUNBUFFERED=1
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings PYTHONIOENCODING=utf-8
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings USE_PYSQLITE=True
az webapp config set -n cohere-chat-app-backend -g openai-chat-app --startup-file="backend/startup.sh"
az webapp deploy --async --type zip --resource-group openai-chat-app --name cohere-chat-app-backend --src-path $PROJECT_DIR/backend-deploy.zip
