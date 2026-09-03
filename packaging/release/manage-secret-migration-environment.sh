#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="plaid-ai/NUV-AGENT"
readonly MIGRATION_ENVIRONMENT="release-secret-migration"
readonly WORKFLOW_PATH=".github/workflows/migrate-release-secrets.yml"
readonly PLATFORM_ADMIN_TEAM_ID="16128529"

die() {
  echo "release-secret migration environment error: $*" >&2
  exit 1
}

authenticated_admin() {
  local login permission membership
  login="$(gh api user --jq .login)"
  permission="$(gh api "repos/${REPOSITORY}/collaborators/${login}/permission" --jq .permission)"
  [ "$permission" = admin ] || die "authenticated user is not a repository administrator"
  membership="$(gh api "teams/${PLATFORM_ADMIN_TEAM_ID}/memberships/${login}" --jq '[.state,.role] | @tsv')"
  case "$membership" in
    $'active\tmember'|$'active\tmaintainer') ;;
    *) die "authenticated user is not an active Platform-Admin member" ;;
  esac
}

setup_environment() {
  [ "$#" -eq 0 ] || die "setup reads the administrator token exclusively from stdin"
  [ ! -t 0 ] || die "pipe the administrator token to setup on stdin"
  local admin_token login permission membership
  IFS= read -r admin_token || true
  [ -n "$admin_token" ] || die "administrator token was not provided on stdin"

  login="$(GH_TOKEN="$admin_token" gh api user --jq .login)"
  permission="$(GH_TOKEN="$admin_token" gh api \
    "repos/${REPOSITORY}/collaborators/${login}/permission" --jq .permission)"
  [ "$permission" = admin ] || die "provided token does not belong to a repository administrator"
  membership="$(GH_TOKEN="$admin_token" gh api \
    "teams/${PLATFORM_ADMIN_TEAM_ID}/memberships/${login}" --jq '[.state,.role] | @tsv')"
  case "$membership" in
    $'active\tmember'|$'active\tmaintainer') ;;
    *) die "provided token does not belong to an active Platform-Admin member" ;;
  esac

  local main_protected workflow_paths
  main_protected="$(GH_TOKEN="$admin_token" gh api \
    "repos/${REPOSITORY}/branches/main" --jq .protected)"
  [ "$main_protected" = true ] || die "main is not protected"
  workflow_paths="$(GH_TOKEN="$admin_token" gh api \
    "repos/${REPOSITORY}/git/trees/main?recursive=1" \
    --jq ".tree[] | select(.path == \"${WORKFLOW_PATH}\") | .path")"
  [ "$workflow_paths" = "$WORKFLOW_PATH" ] \
    || die "merge the one-shot workflow to protected main before setup"

  GH_TOKEN="$admin_token" gh api --method PUT \
    "repos/${REPOSITORY}/environments/${MIGRATION_ENVIRONMENT}" \
    --input - >/dev/null <<'JSON'
{
  "wait_timer": 0,
  "prevent_self_review": false,
  "reviewers": [],
  "deployment_branch_policy": {
    "protected_branches": false,
    "custom_branch_policies": true
  }
}
JSON

  local policies
  policies="$(GH_TOKEN="$admin_token" gh api --paginate \
    "repos/${REPOSITORY}/environments/${MIGRATION_ENVIRONMENT}/deployment-branch-policies?per_page=100" \
    --jq '.branch_policies[] | [.name,.type] | @tsv')"
  if [ -z "$policies" ]; then
    GH_TOKEN="$admin_token" gh api --method POST \
      "repos/${REPOSITORY}/environments/${MIGRATION_ENVIRONMENT}/deployment-branch-policies" \
      -f name=main -f type=branch >/dev/null
  elif [ "$policies" != $'main\tbranch' ]; then
    die "temporary environment has unexpected deployment branch policies"
  fi

  local candidate_environment candidate_policies candidate_secrets
  for candidate_environment in iq9075-candidate-sign iq9075-candidate-stage; do
    GH_TOKEN="$admin_token" gh api --method PUT \
      "repos/${REPOSITORY}/environments/${candidate_environment}" \
      --input - >/dev/null <<'JSON'
{
  "wait_timer": 0,
  "prevent_self_review": false,
  "reviewers": [],
  "deployment_branch_policy": {
    "protected_branches": false,
    "custom_branch_policies": true
  }
}
JSON
    candidate_policies="$(GH_TOKEN="$admin_token" gh api --paginate \
      "repos/${REPOSITORY}/environments/${candidate_environment}/deployment-branch-policies?per_page=100" \
      --jq '.branch_policies[] | [.name,.type] | @tsv')"
    if [ -z "$candidate_policies" ]; then
      GH_TOKEN="$admin_token" gh api --method POST \
        "repos/${REPOSITORY}/environments/${candidate_environment}/deployment-branch-policies" \
        -f name=main -f type=branch >/dev/null
    elif [ "$candidate_policies" != $'main\tbranch' ]; then
      die "candidate environment has unexpected deployment branch policies"
    fi
    candidate_secrets="$(GH_TOKEN="$admin_token" gh api --paginate \
      "repos/${REPOSITORY}/environments/${candidate_environment}/secrets?per_page=100" \
      --jq '.secrets[].name' | LC_ALL=C sort)"
    case "$candidate_environment:$candidate_secrets" in
      iq9075-candidate-sign:|iq9075-candidate-sign:IQ9075_RELEASE_SIGNING_PRIVATE_KEY) ;;
      iq9075-candidate-stage:|$'iq9075-candidate-stage:GCP_PROJECT_ID\nGCP_SA_KEY') ;;
      *) die "candidate environment already contains unexpected secrets" ;;
    esac
  done

  local existing
  existing="$(GH_TOKEN="$admin_token" gh api --paginate \
    "repos/${REPOSITORY}/environments/${MIGRATION_ENVIRONMENT}/secrets?per_page=100" \
    --jq '.secrets[].name' | LC_ALL=C sort)"
  [ -z "$existing" ] || [ "$existing" = RELEASE_SECRET_MIGRATION_ADMIN_TOKEN ] \
    || die "temporary environment already contains unexpected secrets"

  printf '%s' "$admin_token" \
    | GH_TOKEN="$admin_token" gh secret set RELEASE_SECRET_MIGRATION_ADMIN_TOKEN \
        --repo "$REPOSITORY" --env "$MIGRATION_ENVIRONMENT" --app actions >/dev/null
  local actual
  actual="$(GH_TOKEN="$admin_token" gh api --paginate \
    "repos/${REPOSITORY}/environments/${MIGRATION_ENVIRONMENT}/secrets?per_page=100" \
    --jq '.secrets[].name' | LC_ALL=C sort)"
  [ "$actual" = RELEASE_SECRET_MIGRATION_ADMIN_TOKEN ] \
    || die "temporary environment secret metadata is not exact"
  unset admin_token
  echo "release-secret migration environment is ready"
}

cleanup_environment() {
  [ "$#" -eq 0 ] || die "cleanup takes no arguments"
  authenticated_admin

  local workflow_paths
  workflow_paths="$(gh api "repos/${REPOSITORY}/git/trees/main?recursive=1" \
    --jq ".tree[] | select(.path == \"${WORKFLOW_PATH}\") | .path")"
  [ -z "$workflow_paths" ] \
    || die "remove the one-shot workflow from protected main before cleanup"

  local repository_names name
  repository_names="$(gh api --paginate \
    "repos/${REPOSITORY}/actions/secrets?per_page=100" \
    --jq '.secrets[].name' | LC_ALL=C sort)"
  for name in \
    APT_GPG_PASSPHRASE APT_GPG_PRIVATE_KEY GCP_PROJECT_ID GCP_SA_KEY \
    HOMEBREW_TAP_TOKEN IQ9075_RELEASE_PUBLIC_KEYRING_JSON \
    IQ9075_RELEASE_SIGNING_KEY_ID IQ9075_RELEASE_SIGNING_PRIVATE_KEY
  do
    ! grep -Fxq "$name" <<<"$repository_names" \
      || die "forbidden repository secret remains; refusing cleanup"
  done

  gh api --method DELETE \
    "repos/${REPOSITORY}/environments/${MIGRATION_ENVIRONMENT}" >/dev/null
  echo "release-secret migration environment and administrator token were removed"
}

case "${1:-}" in
  setup)
    shift
    setup_environment "$@"
    ;;
  cleanup)
    shift
    cleanup_environment "$@"
    ;;
  *)
    die "usage: gh auth token | $0 setup; $0 cleanup"
    ;;
esac
