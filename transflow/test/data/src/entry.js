const {t} = require('../../../src/transflow');
module.exports = {
  noopProp: 'willnotbetranslated',
  simpleProp: t({default: 'simpleProp.defaultValue'}),
  complexProp: {
    subProp: {
      value: t({default: 'complexProp.subProp.value.defaultValue'})
    }
  },
  arrayProp: [
    {prop: t({default: 'arrayProp.firstItem.prop.defaultValue'})},
    {prop: t({default: 'arrayProp.secondItem.prop.defaultValue'})}
  ],
  arrayOfTranslationsProp: [
    t({default: 'arrayOfTranslationsProp.firstItem.defaultValue'}),
    t({default: 'arrayOfTranslationsProp.secondItem.defaultValue'})
  ],
  willBeTranslatedProp: t({default: 'TOBETRANSLATED'}),
  arrayOfArraysProp: [
    [
      t({default: 'arrayOfArrays(0,0)'}),
      t({default: 'arrayOfArrays(0,1)'})
    ],
    [
      t({default: 'arrayOfArrays(1,0)'}),
      t({default: 'arrayOfArrays(1,1)'})
    ]
  ],
  htmlProp: t({html: true}),
  autoDefaultProp: t('autoDefault')
};
