const {t} = require('../../../src/transflow');

module.exports = [{
  id: 'civil_status',
  label: t({comment: 'civil_status', default: 'État civil'}),
  types: [
    {id: 'fullname', label: t({comment: 'fullname', default: 'Nom complet'})},
    {id: 'firstname', label: t({comment: 'firstname', default: 'Prénom'})},
    {id: 'lastname', label: t({comment: 'lastname', default: 'Patronyme'})},
    {id: 'pseudonym', label: t({comment: 'pseudonym', default: 'Pseudonyme'})},
    {id: 'title', label: t({comment: 'title', default: 'Titre ou civilité'})},
    {id: 'birthday', label: t({comment: 'birthday', default: 'Jour anniversaire'})},
    {id: 'birthdate', label: t({comment: 'birthdate', default: 'Date de naissance'})},
    {id: 'age', label: t({comment: 'age', default: 'Âge'})},
    {id: 'parents_name', label: t({comment: 'parents_name', default: 'Nom des parents'})},
    {id: 'nationality', label: t({comment: 'nationality', default: 'Nationalité'})},
    {id: 'country_of_birth', label: t({comment: 'country_of_birth', default: 'Pays de naissance'})},
    {id: 'place_of_birth', label: t({comment: 'place_of_birth', default: 'Lieu de naissance'})},
    {id: 'marital_status', label: t({comment: 'marital_status', default: 'Statut marital'})},
    {id: 'sex', label: t({comment: 'sex', default: 'Sexe'})}
  ]
}, {
  id: 'mensurations',
  label: t({comment: 'mensurations', default: 'Mensurations'}),
  types: [
    {id: 'silhouette', label: t({comment: 'silhouette', default: 'Silhouette'})},
    {id: 'height', label: t({comment: 'height', default: 'Taille'})},
    {id: 'weight', label: t({comment: 'weight', default: 'Poids'})},
    {id: 'size', label: t({comment: 'size', default: 'Taille de vêtement'})},
    {id: 'footsize', label: t({comment: 'footsize', default: 'Pointure'})},
    {id: 'skin_color', label: t({comment: 'skin_color', default: 'Couleur de peau'})},
    {id: 'skin_type', label: t({comment: 'skin_type', default: 'Type de peau'})},
    {id: 'hair_type', label: t({comment: 'hair_type', default: 'Type de cheveux'})},
    {id: 'hair_color', label: t({comment: 'hair_color', default: 'Couleur de cheveux'})},
    {id: 'eyes', label: t({comment: 'eyes', default: 'Couleur yeux'})},
    {id: 'eye_correction', label: t({comment: 'eye_correction', default: 'Correction occulaire'})}
  ]
}, {
  id: 'ID',
  label: t({comment: 'ID', default: 'Identifiants'}),
  types: [
    {id: 'social_security_id', label: t({comment: 'social_security_id', default: 'Numéro de sécurité sociale'})},
    {id: 'driver_licence_id', label: t({comment: 'driver_licence_id', default: 'Numéro de permis de conduire'})},
    {id: 'identity_number', label: t({comment: 'identity_number', default: 'Numéro d\'identité'})},
    {id: 'passport_number', label: t({comment: 'passport_number', default: 'Numéro de passport'})},
    {
      id: 'student_number',
      label: t({comment: 'student_number', default: 'Numéro d\'Identification National Etudiant (INE)'})
    },
    {
      id: 'car_registration_plate',
      label: t({comment: 'car_registration_plate', default: 'Numéro Plaque d\'Imatriculation'})
    },
    {id: 'bank_account_number', label: t({comment: 'bank_account_number', default: 'Numéro de compte en banque'})},
    {id: 'other_id', label: t({comment: 'other_id', default: 'Autre identifiant'})}
  ]
}, {
  id: 'online_ID',
  label: t({comment: 'online_ID', default: 'Identifiants en ligne'}),
  types: [
    {id: 'google', label: t({comment: 'google', default: 'Google'})},
    {id: 'yahoo', label: t({comment: 'yahoo', default: 'Yahoo'})},
    {id: 'facebook', label: t({comment: 'facebook', default: 'Facebook'})},
    {id: 'linkedin', label: t({comment: 'linkedin', default: 'Linkedin'})},
    {
      id: 'other_platform_id',
      label: t({comment: 'other_platform_id', default: 'Autre identifiant de plateforme en ligne'})
    },
    {
      id: 'social_login_identifier',
      label: t({comment: 'social_login_identifier', default: 'Identifiant de réseau social'})
    }
  ]
}, {
  id: 'contact_information',
  label: t({comment: 'contact_information', default: 'Coordonnées'}),
  types: [
    {id: 'phone_number', label: t({comment: 'phone_number', default: 'Numéro de téléphone'})},
    {id: 'email', label: t({comment: 'email', default: 'Adresse email'})},
    {id: 'professional_email', label: t({comment: 'professional_email', default: 'Adresse email professionnelle'})},
    {id: 'mobile', label: t({comment: 'mobile', default: 'Numéro de téléphone portable'})},
    {id: 'fax', label: t({comment: 'fax', default: 'Numéro de fax'})},
    {id: 'street_address', label: t({comment: 'street_address', default: 'Adresse'})},
    {id: 'business_address', label: t({comment: 'business_address', default: 'Adresse Professionnelle'})},
    {id: 'city', label: t({comment: 'city', default: 'Ville'})},
    {id: 'postal_code', label: t({comment: 'postal_code', default: 'Code Postal'})},
    {id: 'province', label: t({comment: 'province', default: 'Région ou Province'})},
    {id: 'country', label: t({comment: 'country', default: 'Pays'})},
    {id: 'geoposition', label: t({comment: 'geoposition', default: 'Position (Lat/Lon)'})}
  ]
}, {
  id: 'usage_history',
  label: t({comment: 'usage_history', default: 'Historique d\'usage'}),
  types: [
    {id: 'chat_log', label: t({comment: 'chat_log', default: 'Historique de conversation'})},
    {id: 'browsing_history', label: t({comment: 'browsing_history', default: 'Historique de navigation'})},
    {id: 'purchase_history', label: t({comment: 'purchase_history', default: 'Historique d\'achats'})},
    {id: 'phone_call_log', label: t({comment: 'phone_call_log', default: 'Historique d\'appel téléphonique'})}
  ]
}, {
  id: 'business',
  label: t({comment: 'business', default: 'Informations professionnelles'}),
  types: [
    {id: 'job_position', label: t({comment: 'job_position', default: 'Poste occupé dans l\'entreprise'})},
    {id: 'company_name', label: t({comment: 'company_name', default: 'Nom de l\'entreprise'})},
    {id: 'job_history', label: t({comment: 'job_history', default: 'Emplois précédents'})},
    {
      id: 'company_identification_number',
      label: t({
        comment: 'company_identification_number',
        default: 'Numéro d\'identification de l\'entreprise (SIRET)'
      })
    },
    {
      id: 'company_vat_number',
      label: t({comment: 'company_vat_number', default: 'Numéro de TVA de l\'entreprise'})
    }
  ]
}, {
  id: 'interest',
  label: t({comment: 'interest', default: 'Centres d\'intérêts'}),
  types: [
    {id: 'hobbies', label: t({comment: 'hobbies', default: 'Loisirs'})},
    {id: 'intelligence', label: t({comment: 'intelligence', default: 'Veille'})},
    {id: 'sports', label: t({comment: 'sports', default: 'Sport'})},
    {id: 'arts', label: t({comment: 'arts', default: 'Pratique artistique'})}
  ]
}, {
  id: 'technical_information',
  label: t({comment: 'technical_information', default: 'Informations Techniques'}),
  types: [
    {id: 'ip_address', label: t({comment: 'ip_address', default: 'Adresse IP'})},
    {id: 'web_browser', label: t({comment: 'web_browser', default: 'Navigateur'})},
    {id: 'operating_system', label: t({comment: 'operating_system', default: 'Système d\'exploitation'})}
  ]
}, {
  id: 'image',
  label: t({comment: 'image', default: 'Image'}),
  types: [
    {id: 'photo', label: t({comment: 'photo', default: 'Photo'})},
    {id: 'video', label: t({comment: 'video', default: 'Vidéo'})}
  ]
}];
