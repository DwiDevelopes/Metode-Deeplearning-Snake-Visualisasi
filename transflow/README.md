# Transflow

[![Build Status](https://travis-ci.org/agilitation/transflow.svg?branch=master)](https://travis-ci.org/agilitation/transflow)
[![Coverage Status](https://coveralls.io/repos/github/agilitation/transflow/badge.svg?branch=master)](https://coveralls.io/github/agilitation/transflow?branch=master)

Translfow is a Javascript Translation Workflow

## Paradigm

1. Define your data in pure JS objects
2. Transflow generates commented PO files
3. Edit or share the PO file for editing
4. Run Transflow to compile them to JSON

## Usage

### Installation
```bash
npm install transflow
```

### Config
```javascript
module.exports = {
    // mandatory props
    entries: {
        webapp: './src/strings/webapp.js',
        landing: './src/strings/landing.js'
    },
    locales: ['en', 'fr'],
    // optional props
    localeDir: './locale',
    genDir: './gen',
    defaultLocale: 'en',
    baseDir: require('path').join(__dirname, '.')
};
```

### Run

Can be run in CLI and in Node. The API is synchronous, but it may change in the future.
