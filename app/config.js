// Where the app sends ratings. Leave both empty to keep ratings only in the browser.
// The publishable key is meant to be public (it can only add votes, see supabase/schema.sql).
// Never put the secret key here.
window.APP_CONFIG = {
  supabaseUrl: "",        // e.g. "https://abcdefgh.supabase.co"
  supabaseKey: "",        // the "publishable" key, starts with "sb_publishable_"
};
