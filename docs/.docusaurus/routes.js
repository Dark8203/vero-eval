import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs/__docusaurus/debug',
    component: ComponentCreator('/docs/__docusaurus/debug', 'e58'),
    exact: true
  },
  {
    path: '/docs/__docusaurus/debug/config',
    component: ComponentCreator('/docs/__docusaurus/debug/config', '2ce'),
    exact: true
  },
  {
    path: '/docs/__docusaurus/debug/content',
    component: ComponentCreator('/docs/__docusaurus/debug/content', '11b'),
    exact: true
  },
  {
    path: '/docs/__docusaurus/debug/globalData',
    component: ComponentCreator('/docs/__docusaurus/debug/globalData', 'f13'),
    exact: true
  },
  {
    path: '/docs/__docusaurus/debug/metadata',
    component: ComponentCreator('/docs/__docusaurus/debug/metadata', 'bff'),
    exact: true
  },
  {
    path: '/docs/__docusaurus/debug/registry',
    component: ComponentCreator('/docs/__docusaurus/debug/registry', '830'),
    exact: true
  },
  {
    path: '/docs/__docusaurus/debug/routes',
    component: ComponentCreator('/docs/__docusaurus/debug/routes', '13e'),
    exact: true
  },
  {
    path: '/docs/',
    component: ComponentCreator('/docs/', 'ec8'),
    routes: [
      {
        path: '/docs/',
        component: ComponentCreator('/docs/', '093'),
        routes: [
          {
            path: '/docs/',
            component: ComponentCreator('/docs/', '262'),
            routes: [
              {
                path: '/docs/getting-started/project-structure',
                component: ComponentCreator('/docs/getting-started/project-structure', '2a1'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '165'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/getting-started/tutorials/end-to-end-eval',
                component: ComponentCreator('/docs/getting-started/tutorials/end-to-end-eval', '9e2'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/alignscore',
                component: ComponentCreator('/docs/metrics/generation/alignscore', '2b4'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/bartscore',
                component: ComponentCreator('/docs/metrics/generation/bartscore', 'e64'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/bertscore',
                component: ComponentCreator('/docs/metrics/generation/bertscore', '1a2'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/bleurtscore',
                component: ComponentCreator('/docs/metrics/generation/bleurtscore', 'f03'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/geval',
                component: ComponentCreator('/docs/metrics/generation/geval', 'eab'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/rougescore',
                component: ComponentCreator('/docs/metrics/generation/rougescore', '689'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/generation/semscore',
                component: ComponentCreator('/docs/metrics/generation/semscore', '5e8'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/overview',
                component: ComponentCreator('/docs/metrics/overview', 'eee'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/reranker/cumulative-ndcg',
                component: ComponentCreator('/docs/metrics/reranker/cumulative-ndcg', 'a1e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/reranker/meanap',
                component: ComponentCreator('/docs/metrics/reranker/meanap', '081'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/reranker/meanrr',
                component: ComponentCreator('/docs/metrics/reranker/meanrr', '3e6'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/reranker/rerank-ndcg',
                component: ComponentCreator('/docs/metrics/reranker/rerank-ndcg', '3c1'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/retriever/citationscore',
                component: ComponentCreator('/docs/metrics/retriever/citationscore', 'd94'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/retriever/precisionscore',
                component: ComponentCreator('/docs/metrics/retriever/precisionscore', 'c81'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/retriever/recallscore',
                component: ComponentCreator('/docs/metrics/retriever/recallscore', '891'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/metrics/retriever/sufficiencyscore',
                component: ComponentCreator('/docs/metrics/retriever/sufficiencyscore', '696'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', 'be8'),
                exact: true,
                sidebar: "docs"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
