// docusaurus.config.js
/** @type {import('@docusaurus/types').Config} */
module.exports = {
  title: 'AI Evaluation Framework',
  tagline: 'Evaluate AI pipelines with tracing, logging and extensive metrics',
  url: 'https://vero.co.in',          // change
  baseUrl: '/docs/',                  // set trailing slash for correct routing
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  projectName: 'vero-website',
  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-org/your-repo-name/edit/main/docs/',
          routeBasePath: '/', // serve docs at site root of the Docusaurus build
          showLastUpdateAuthor: false,
          showLastUpdateTime: false
        },
        blog: false, // disable blog
        pages: false, // disable custom pages (like homepage)
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
    themeConfig: {
    colorMode: {
      defaultMode: 'dark',            // <- set dark as the default
      disableSwitch: false,           // <- keep the toggle so user can switch
      respectPrefersColorScheme: false // <- ignore the user’s OS theme preference
    },
},
};
