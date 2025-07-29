import re

certidao_protesto = re.compile(r'((certid[aã]o).{0,100}(protesto))',flags=re.IGNORECASE|re.S)
integralizado = re.compile(r'((capital.{0,20}integralizado)|(patrim[oô]nio.{0,20}integralizado))', flags=re.IGNORECASE|re.S)
idoneidade_financeira = re.compile('((atesta[do]*)|(certid[ãa]o)|(declara[çc]*[ãa]*[o]*))(.{0,70}((i[ni]*doneidade)).{0,20}((financeira)|(banc[áa]ria)))', flags=re.IGNORECASE|re.S)
comprovante_localizacao = re.compile('(((alvar[aá])|(comprov[mnteçcãao]*)).{0,20}((localiza[çc][ãa]o)|(funcionamento)))',flags=re.IGNORECASE|re.S)
n_min_max_limitacao_atestados = re.compile(r'((((dois|duas)|(tr[êe]s)|(quatro)|(cinco)).{0,10}((atestado[s]?)|(certid[aãoões]*))).{0,20}((capacidade t[eé]cnica)|(qualifica[cç][aã]o t[eé]cnica)))', flags=re.IGNORECASE|re.S)
certificado_boas_praticas = re.compile(r'(((certificado[s]?)|(atestado[s]?)|(certid[aã]o)).{0,20}(boa[s]?.{0,5}pr[aá]tica[s]?))', flags=re.IGNORECASE|re.S)
licenca_ambiental = re.compile(r'((licen[cç]a).{0,20}(ambiental))', flags=re.IGNORECASE|re.S)

garantia_cs_ou_pl = re.compile(r'(garantia)(.{0,20}((patrim[oô]nio.{0,10}l[íi]qu[ií]do)|(capital.{0,10}social)))|((capital).{0,10}(social))(.{0,10}(patrim[oô]nio.{0,10}(l[íi]qu[ií]do).{0,10}(m[ií]n[ií]mo)))', flags=re.IGNORECASE|re.S)
carta_credenciamento = re.compile(r'(((comprova[cçan][tãa][eo]).{0,15}(revenda))|(credenciamento.{0,10}fabricante)|(carta.{0,10}solidariedade))', flags=re.IGNORECASE|re.S)
filiacao_abav_iata = re.compile(r'((((associa[çc][aã]o)|(empresa[s])|(sindicato))(.{0,30}((turismo)|(transporte)|(viage[mns]+)|(aerovi[áa]ria[s]?))))|((international)(.{0,10}((air).{0,10}(transport).{0,10}(association))))|(((\s)|(\n))embratur(\s))|(((\s)|(\n))iata(\s))|(((\s)|(\n))abav(\s))|(((\s)|(\n))sindetur(\s))|(((\s)|(\n))snea(\s)))', flags=re.IGNORECASE|re.S)
contratacao_visto_registro_profissional = re.compile(r'(contrata[çc][ãa]o.{1,15})(((visto).{0,20}([cC]onselho|CREA|CAU|entidade.{0,10}profissional|registro))|((caso).{0,20}(empresa[s]?.{0,10}n[ãa]o.{0,10}sediada[s]?|licitante[s]?.{0,10}n[aã]o.{0,10}sediado[s]?)))',re.S)
temp_exp_profissional = re.compile(r'(((tempo).{0,20}((experi[eê]ncia)|(exerc[ií]cio.{0,10}profissional)|(registro.{0,10}conselho)|(forma[cç][aã]o.{0,10}acad[eê]mica)))|(experi[êe]ncia.{0,10}ano)|(ano[s]?.{0,10}experi[eê]ncia))', flags=re.IGNORECASE|re.S)
atestado_regularizacao = re.compile(r'(((atestado).{0,50}((regulariza[cç][aã]o)|(regularizado)|(regularizar)))|((t[eé]cnico operacional).{0,50}((regulariza[cç][aã]o)|(regularizado)|(regularizar)))|((capacidade t[eé]cnica).{0,50}((regulariza[cç][aã]o)|(regularizar)|(regularizado))))', flags=re.IGNORECASE|re.S)
atestado_carga = re.compile(r'(((atestado).{0,50}((carregamento)|(carga)|(carregado)|(carregar)))|((t[eé]cnico operacional).{0,50}((carregamento)|(carga)|(carregado)|(carregar)))|((capacidade t[eé]cnica).{0,50}((carregamento)|(carga)|(carregado)|(carregar))))', flags=re.IGNORECASE|re.S)


expressoes = {'certidao_protesto': certidao_protesto, 
              'integralizado': integralizado,
              'idoneidade_financeira': idoneidade_financeira,
              'comprovante_localizacao': comprovante_localizacao,
              'n_min_max_limitacao_atestados': n_min_max_limitacao_atestados,
              'certificado_boas_praticas': certificado_boas_praticas,
              'licenca_ambiental': licenca_ambiental,
              'garantia_cs_ou_pl': garantia_cs_ou_pl,
              'carta_credenciamento': carta_credenciamento,
              'filiacao_abav_iata': filiacao_abav_iata,
              'temp_exp_profissional': temp_exp_profissional,
              'atestado_regularizacao': atestado_regularizacao,
              'atestado_carga': atestado_carga
             }