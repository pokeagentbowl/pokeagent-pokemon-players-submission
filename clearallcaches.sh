rm -rf .pokeagent_cache/*
rm -rf navigation_caches/*
rm -f startup_cache/dialogues.json
rm -f startup_cache/actions.json
rm -rf llm_logs/*
rm -f submission.log
sed -E -i 's/\[[[:space:]]*[xX][[:space:]]*\]/[ ]/g' startup_cache/startup_objectives.json