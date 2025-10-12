// sidebars.js
module.exports = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/quickstart',
        'getting-started/project-structure',
        {
          type: 'category',
          label: 'Tutorials',
          items: [
            'getting-started/tutorials/end-to-end-eval',
          ]
        }
      ]
    },
    {
      type: 'category',
      label: 'Metrics',
      items: [
        'metrics/overview',
        {
          type: 'category',
          label: 'Generation Metrics',
          items: [
            'metrics/generation/bertscore',
            'metrics/generation/bartscore',
            'metrics/generation/bleurtscore',
            'metrics/generation/rougescore',
            'metrics/generation/semscore',
            'metrics/generation/alignscore',
            'metrics/generation/geval',
          ]
        },
        {
          type: 'category',
          label: 'Retriever Metrics',
          items: [
            'metrics/retriever/recallscore',
            'metrics/retriever/precisionscore',
            'metrics/retriever/citationscore',
            'metrics/retriever/sufficiencyscore',
          ]
        },
        {
          type: 'category',
          label: 'Reranker Metrics',
          items: [
            'metrics/reranker/meanrr',
            'metrics/reranker/meanap',
            'metrics/reranker/rerank-ndcg',
            'metrics/reranker/cumulative-ndcg',
          ]
        },
      ]
    },
  ],
};
