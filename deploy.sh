#!/bin/sh

PROJECT_DIR=$(pwd)
BACKEND_DIR=src
BACKEND_BUILD_DIR=deploy/backend
FRONTEND_BUILD_DIR=tmp/frontend

# build frontend static files
rm -rf $FRONTEND_BUILD_DIR
cat src/interfaces/coral_web/.env.production > src/interfaces/coral_web/.env
docker compose run --rm --build frontend npm run build;
# cp -r tmp/frontend/output src/backend/ui/output

# return
rm -rf $BACKEND_BUILD_DIR
mkdir -p $BACKEND_BUILD_DIR
cp pyproject.toml $BACKEND_BUILD_DIR/
cp poetry.lock $BACKEND_BUILD_DIR/
cp -r $BACKEND_DIR/backend $BACKEND_BUILD_DIR/
cp -r $BACKEND_DIR/community $BACKEND_BUILD_DIR/
rm -rf $BACKEND_BUILD_DIR/backend/ui
cp -r $FRONTEND_BUILD_DIR $BACKEND_BUILD_DIR/backend/ui
rm -rf $BACKEND_BUILD_DIR/backend/data
mkdir -p $BACKEND_BUILD_DIR/backend/data
source .deployenv
echo "DATABASE_URL=$DATABASE_URL" > $BACKEND_BUILD_DIR/.env
echo "USE_COMMUNITY_FEATURES=$USE_COMMUNITY_FEATURES" >> $BACKEND_BUILD_DIR/.env
echo "COHERE_API_KEY=$COHERE_API_KEY" >> $BACKEND_BUILD_DIR/.env
echo "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION" >> $BACKEND_BUILD_DIR/.env
echo "AZURE_OPENAI_RESOURCE=$AZURE_OPENAI_RESOURCE" >> $BACKEND_BUILD_DIR/.env
echo "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" >> $BACKEND_BUILD_DIR/.env
echo "AZURE_OPENAI_MODEL=$AZURE_OPENAI_MODEL" >> $BACKEND_BUILD_DIR/.env
docker compose run --rm --build backend bash -c "pip freeze >> /workspace/tmp/requirement.txt" 
rm -f $PROJECT_DIR/backend-deploy.zip
cd $BACKEND_BUILD_DIR
zip -vr $PROJECT_DIR/backend-deploy.zip * .*
cd $PROJECT_DIR
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings PYTHONDONTWRITEBYTECODE=1
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings PYTHONUNBUFFERED=1
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings PYTHONIOENCODING=utf-8
az webapp config appsettings set -n cohere-chat-app-backend -g openai-chat-app --settings USE_PYSQLITE=True
az webapp config set -n cohere-chat-app-backend -g openai-chat-app --startup-file="backend/startup.sh"
az webapp deploy --async --type zip --resource-group openai-chat-app --name cohere-chat-app-backend --src-path $PROJECT_DIR/backend-deploy.zip
az webapp restart -n cohere-chat-app-backend -g openai-chat-app