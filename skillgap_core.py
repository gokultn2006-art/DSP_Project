"""Core components: taxonomy, role rules, NLP skill extraction, data loading."""
import ast, re, os, pickle
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

T = {}
def add(cat, name, *forms):
    T[name] = (cat, [name.lower()] + [f.lower() for f in forms])

P, D, W, C, S, Q, DB = "Programming", "Data & ML", "Web & Mobile", "Cloud & DevOps", "Security & Network", "Testing & Process", "Databases"
for n, f in [("Python", ()), ("Java", ("core java", "java se", "j2ee")), ("JavaScript", ("js", "es6", "ecmascript")), ("TypeScript", ()),
             ("C++", ("cpp",)), ("C#", ("c sharp", "csharp")), ("Go", ("golang",)), ("Rust", ()), ("Scala", ()), ("Kotlin", ()), ("Swift", ()),
             ("PHP", ()), ("Ruby", ("ruby on rails", "rails")), ("MATLAB", ()), ("Bash", ("shell scripting", "shell script", "shell scripts")),
             ("PowerShell", ()), ("SQL", ("t-sql", "pl/sql", "pl-sql", "structured query language")), ("R", ()), ("SAS", ())]:
    add(P, n, *f)
for n, f in [("Machine Learning", ("ml", "machine-learning")), ("Deep Learning", ("neural networks", "neural network")),
             ("NLP", ("natural language processing", "text mining")), ("Computer Vision", ("image processing", "opencv")),
             ("Generative AI", ("genai", "gen ai", "generative artificial intelligence")), ("LLMs", ("large language models", "llm", "chatgpt", "langchain")),
             ("Artificial Intelligence", ("ai",)), ("Statistics", ("statistical analysis", "statistical modeling", "statistical modelling", "statistical")),
             ("Data Analysis", ("data analytics", "data analyst")), ("Data Visualization", ("data visualisation", "data viz", "dashboards", "dashboarding")),
             ("Data Mining", ()), ("Predictive Modeling", ("predictive analytics", "predictive modelling", "predictive models")),
             ("Pandas", ()), ("NumPy", ()), ("Scikit-learn", ("sklearn", "scikit learn")), ("TensorFlow", ()), ("PyTorch", ("torch",)), ("Keras", ()),
             ("Matplotlib", ()), ("Spark", ("apache spark", "pyspark", "spark sql")), ("Hadoop", ("hdfs", "mapreduce", "hive")), ("Kafka", ("apache kafka",)),
             ("Airflow", ("apache airflow",)), ("ETL", ("data pipelines", "data pipeline", "elt")), ("Data Warehousing", ("data warehouse", "data warehouses", "redshift", "bigquery")),
             ("Data Modeling", ("data modelling", "dimensional modeling")), ("Tableau", ()), ("Power BI", ("powerbi", "power-bi")), ("Excel", ("microsoft excel", "ms excel", "advanced excel", "excel spreadsheets")),
             ("Snowflake", ()), ("Databricks", ()), ("Looker", ()), ("Big Data", ()), ("A/B Testing", ("ab testing", "a/b tests", "split testing")),
             ("Time Series", ("time-series", "forecasting")), ("Feature Engineering", ()), ("Linear Regression", ("logistic regression", "regression analysis")),
             ("MLOps", ("model deployment", "mlflow", "kubeflow")), ("Reinforcement Learning", ())]:
    add(D, n, *f)
for n, f in [("MySQL", ()), ("PostgreSQL", ("postgres",)), ("MongoDB", ("mongo db",)), ("Oracle", ("oracle database", "oracle db")),
             ("SQL Server", ("mssql", "ms sql", "microsoft sql server")), ("Redis", ()), ("NoSQL", ()), ("Elasticsearch", ("elastic search",)),
             ("Cassandra", ()), ("DynamoDB", ()), ("Database Administration", ("database management", "dba", "database tuning", "backup and recovery"))]:
    add(DB, n, *f)
for n, f in [("HTML", ("html5",)), ("CSS", ("css3", "sass", "scss")), ("React", ("reactjs", "react.js", "react js")), ("Angular", ("angularjs", "angular js")),
             ("Vue.js", ("vue", "vuejs", "vue js")), ("Node.js", ("nodejs", "node js", "node")), ("Django", ()), ("Flask", ()), ("Spring Boot", ("springboot", "spring framework")),
             (".NET", ("dotnet", ".net core", "asp.net")), ("REST APIs", ("rest api", "restful", "restful api", "restful apis", "rest services", "web services", "api development")),
             ("GraphQL", ()), ("Microservices", ("micro services", "microservice")), ("Android", ("android studio", "android development")),
             ("iOS", ("ios development", "xcode", "swiftui")), ("Flutter", ()), ("React Native", ()), ("jQuery", ()), ("Bootstrap", ()), ("Firebase", ())]:
    add(W, n, *f)
for n, f in [("AWS", ("amazon web services", "ec2", "s3", "aws lambda")), ("Azure", ("microsoft azure", "azure devops")), ("GCP", ("google cloud", "google cloud platform")),
             ("Cloud Computing", ("cloud infrastructure", "cloud services", "cloud platforms")), ("Docker", ("containerization", "containers")),
             ("Kubernetes", ("k8s",)), ("Terraform", ("infrastructure as code", "iac", "cloudformation")), ("Ansible", ("puppet", "chef")),
             ("Jenkins", ()), ("CI/CD", ("cicd", "ci cd", "ci/cd pipelines", "continuous integration", "continuous delivery", "continuous deployment", "gitlab ci", "github actions")),
             ("Git", ("github", "gitlab", "bitbucket", "version control")), ("Linux", ("unix", "red hat", "rhel", "ubuntu")), ("Prometheus", ("grafana", "datadog", "observability")),
             ("Windows Server", ("active directory", "group policy")), ("VMware", ("vsphere", "virtualization", "hyper-v")),
             ("Networking", ("tcp/ip", "lan/wan", "routing and switching", "cisco", "ccna", "network administration", "dns", "dhcp", "vpn")),
             ("Site Reliability", ("sre", "incident management", "monitoring and alerting"))]:
    add(C, n, *f)
for n, f in [("Network Security", ("firewalls", "firewall", "intrusion detection", "ids/ips")), ("Penetration Testing", ("pentesting", "pen testing", "ethical hacking", "red team")),
             ("SIEM", ("splunk", "qradar", "sentinel", "security monitoring")), ("Cryptography", ("encryption", "pki")), ("OWASP", ("owasp top 10", "application security", "appsec")),
             ("Incident Response", ("soc", "threat hunting", "digital forensics")), ("Vulnerability Management", ("vulnerability assessment", "vulnerability scanning", "nessus")),
             ("IAM", ("identity and access management", "access management", "sso")), ("Risk Assessment", ("risk management", "risk analysis")),
             ("Compliance", ("iso 27001", "nist", "soc 2", "pci dss", "hipaa", "gdpr")), ("Cybersecurity", ("cyber security", "information security", "infosec"))]:
    add(S, n, *f)
for n, f in [("Selenium", ("selenium webdriver",)), ("Test Automation", ("automated testing", "automation testing", "test automation framework", "automation framework")),
             ("JUnit", ("testng", "pytest", "unit testing", "unit tests")), ("Cypress", ("playwright",)), ("Postman", ("api testing", "soapui")),
             ("JIRA", ("confluence",)), ("Agile", ("scrum", "kanban", "sprint")), ("Manual Testing", ("test cases", "test plans", "regression testing", "functional testing", "uat")),
             ("Performance Testing", ("jmeter", "load testing", "loadrunner")), ("Project Management", ("project manager", "pmp", "program management")),
             ("Business Analysis", ("requirements gathering", "requirement gathering", "business requirements", "user stories", "functional requirements")),
             ("Stakeholder Management", ("stakeholder engagement", "stakeholders")), ("Technical Support", ("help desk", "service desk", "troubleshooting", "ticketing")),
             ("System Design", ("software architecture", "design patterns", "object-oriented design", "oop", "object oriented programming")),
             ("Data Structures", ("algorithms", "data structures and algorithms")), ("SDLC", ("software development life cycle", "software development lifecycle"))]:
    add(Q, n, *f)

SKILLS = sorted(T)
S_N = len(SKILLS)
SIDX = {s: i for i, s in enumerate(SKILLS)}
CATEGORY = {s: T[s][0] for s in SKILLS}

CS_PATTERNS = {
    "R": r"(?<![A-Za-z0-9.&/])R(?![A-Za-z0-9&+#.\-'])(?!\s*&\s*D)",
    "SAS": r"(?<![A-Za-z0-9])SAS(?![A-Za-z0-9])",
    "React": r"(?<![A-Za-z0-9])React(?![A-Za-z0-9.])(?!\s+(?:to|quickly|and|appropriately|effectively))",
    "Excel": r"(?<![A-Za-z0-9])Excel(?![A-Za-z0-9])(?!\s+(?:in|at|under|as|with a|within|and|on))",
    "Spark": r"(?<![A-Za-z0-9])Spark(?![A-Za-z0-9])(?!\s+(?:creativity|interest|innovation|ideas|passion))",
    "Machine Learning": r"(?<![A-Za-z0-9])ML(?![A-Za-z0-9])",
    "Artificial Intelligence": r"(?<![A-Za-z0-9])AI(?![A-Za-z0-9])",
    "Go": r"(?<![A-Za-z0-9])Golang(?![A-Za-z0-9])|(?<![A-Za-z0-9])Go\s+(?:lang|programming|language)(?![A-Za-z0-9])",
    "Swift": r"(?<![A-Za-z0-9])Swift(?![A-Za-z0-9])(?!\s+(?:and|response|action|resolution))",
    "Node.js": r"(?<![A-Za-z0-9])Node(?:\.?js)?(?![A-Za-z0-9])",
}
AMBIG = set(CS_PATTERNS) | {"R", "SAS", "ai", "ml", "node", "swift", "go"}
ALL_FORMS = {}                       # lower surface form -> canonical (case-insensitive forms only)
for sk, (_, fs) in T.items():
    for f in fs:
        if sk in ("R", "SAS") or f in ("ai", "ml", "node"):  # handled by case-sensitive patterns
            continue
        if sk in ("React", "Excel", "Spark", "Swift", "Go") and f == sk.lower():
            continue
        ALL_FORMS[f] = sk
CANON_FORMS = {sk.lower(): sk for sk in SKILLS if sk.lower() in ALL_FORMS}


def _pat(forms):
    fs = sorted(forms, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9+#])(" + "|".join(re.escape(f) for f in fs) + r")(?![a-z0-9+#])")


PAT_EXACT = _pat(list(CANON_FORMS))
PAT_ALIAS = _pat(list(ALL_FORMS))
CS_COMPILED = {k: re.compile(v) for k, v in CS_PATTERNS.items()}
PREFERRED = re.compile(r"(preferred|nice to have|nice-to-have|a plus|is a plus|bonus|desirable|ideally|advantageous|good to have|would be an asset|familiarity)", re.I)
SEG_SPLIT = re.compile(r"[\n•;]|(?<=[.!?])\s+")
LIST_SPECIAL = {"r": "R", "sas": "SAS", "react": "React", "excel": "Excel", "spark": "Spark", "go": "Go", "swift": "Swift", "ai": "Artificial Intelligence",
                "ml": "Machine Learning", "node": "Node.js", "c": None, "sql": "SQL"}
FUZZ_KEYS = [k for k in ALL_FORMS if len(k) >= 6]


def extract_text(text, variant="alias", with_strength=False):
    """Skills mentioned in free text. variant: 'exact' (canonical names) or 'alias' (+ aliases + context patterns).
    Returns set, or dict skill->strength (1.0 required, 0.5 preferred-only) when with_strength."""
    out = {}
    pat = PAT_EXACT if variant == "exact" else PAT_ALIAS
    forms = CANON_FORMS if variant == "exact" else ALL_FORMS
    for seg in SEG_SPLIT.split(text):
        if not seg.strip():
            continue
        st = 0.5 if PREFERRED.search(seg) else 1.0
        low = seg.lower()
        for m in pat.findall(low):
            sk = forms[m]
            out[sk] = max(out.get(sk, 0), st)
        if variant != "exact":
            for sk, cp in CS_COMPILED.items():
                if cp.search(seg):
                    out[sk] = max(out.get(sk, 0), st)
    return out if with_strength else set(out)


_fz = {}
def map_item(item, variant):
    """Map one resume skill-list item to the taxonomy. variant: exact | alias | fuzzy."""
    low = item.strip().lower()
    if not low:
        return None
    if low in CANON_FORMS:
        return CANON_FORMS[low]
    if variant == "exact":
        return None
    if low in ALL_FORMS:
        return ALL_FORMS[low]
    if low in LIST_SPECIAL:
        return LIST_SPECIAL[low]
    hit = extract_text(item, "alias")
    if len(hit) == 1:
        return next(iter(hit))
    if variant == "fuzzy" and len(low) >= 6:
        if low not in _fz:
            m = process.extractOne(low, FUZZ_KEYS, scorer=fuzz.ratio, score_cutoff=88)
            _fz[low] = ALL_FORMS[m[0]] if m else None
        return _fz[low]
    return None


def vec(skillset):
    v = np.zeros(S_N)
    for s in skillset:
        v[SIDX[s]] = 1
    return v


# --------------------------------------------------------------------------- #
# Roles (title rules; first match wins)
# --------------------------------------------------------------------------- #
ROLE_RULES = [
 ("Data Scientist", r"data scien"),
 ("ML/AI Engineer", r"machine learning|\bml engineer|\bai engineer|artificial intelligence|deep learning|\bnlp\b|generative ai"),
 ("Data Analyst", r"data analyst|business intelligence|bi analyst|analytics analyst|reporting analyst|bi developer"),
 ("Data Engineer", r"data engineer|\betl\b|big data|data warehouse|analytics engineer"),
 ("Business Analyst", r"business analyst|systems analyst|product analyst|it analyst"),
 ("Full Stack Developer", r"full.?stack"),
 ("Frontend Developer", r"front.?end|ui developer|react developer|angular|web developer|ui engineer"),
 ("Mobile Developer", r"android|\bios\b|mobile developer|mobile engineer|flutter|react native"),
 ("Cybersecurity Analyst", r"cyber|security analyst|security engineer|information security|soc analyst|penetration|infosec"),
 ("DevOps/Cloud Engineer", r"devops|cloud engineer|site reliability|\bsre\b|platform engineer|cloud architect|infrastructure engineer|cloud developer"),
 ("QA/Test Engineer", r"\bqa\b|quality assurance|test engineer|sdet|test automation|software tester|\btester\b"),
 ("Database Administrator", r"database admin|\bdba\b|database engineer|sql developer|database developer"),
 ("Network/Systems Admin", r"network engineer|network admin|system admin|systems admin|sysadmin|it support|help desk|network support|desktop support|systems engineer"),
 ("Software Engineer", r"software engineer|software developer|application developer|programmer|back.?end|java developer|python developer|\.net developer|golang|developer"),
]
ROLE_NAMES = [r for r, _ in ROLE_RULES]
R_N = len(ROLE_NAMES)
_ROLE_RE = [(i, re.compile(p)) for i, (_, p) in enumerate(ROLE_RULES)]


def title_to_role(title):
    t = str(title).lower()
    for i, rx in _ROLE_RE:
        if rx.search(t):
            return i
    return -1



# Data loading
def load_postings(path, cache_dir):
    cp = os.path.join(cache_dir, "postings_extracted.pkl")
    if os.path.exists(cp):
        return pickle.load(open(cp, "rb"))
    use = ["job_id", "title", "description", "skills_desc", "formatted_experience_level", "normalized_salary", "views", "applies", "listed_time", "work_type"]
    df = pd.read_csv(path, usecols=use)
    df["role"] = df["title"].map(title_to_role)
    df = df[df["role"] >= 0].reset_index(drop=True)
    # the title is NOT used for extraction or text features, because the role label is derived from it
    df["text"] = df["description"].fillna("") + ". " + df["skills_desc"].fillna("")
    strength = np.zeros((len(df), S_N), dtype=np.float32)
    exact = np.zeros((len(df), S_N), dtype=np.float32)
    for i, txt in enumerate(df["text"].values):
        for s, v in extract_text(txt, "alias", with_strength=True).items():
            strength[i, SIDX[s]] = v
        for s in extract_text(txt, "exact"):
            exact[i, SIDX[s]] = 1
    res = (df.drop(columns=["description", "skills_desc"]), strength, exact)  # keeps df["text"] for the TF-IDF baselines
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(cp, "wb"))
    return res


def _lst(s):
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def load_resumes(path):
    r = pd.read_csv(path)
    r.columns = [c.replace("﻿", "") for c in r.columns]
    r["skill_items"] = r["skills"].map(lambda s: _lst(s) if isinstance(s, str) else [])
    r["pos_list"] = r["positions"].map(lambda s: _lst(s) if isinstance(s, str) else [])
    r["rel_list"] = r["related_skils_in_job"].map(lambda s: [x for l in _lst(s) for x in (l if isinstance(l, list) else [l])] if isinstance(s, str) else [])
    r["cert"] = r["certification_skills"].fillna("").astype(str)
    uniq = r.drop_duplicates(subset=["skills", "career_objective", "positions"]).reset_index(drop=True)
    uniq["rid"] = range(len(uniq))
    key = r["skills"].fillna("") + "|" + r["career_objective"].fillna("") + "|" + r["positions"].fillna("")
    kmap = {k: i for i, k in enumerate((uniq["skills"].fillna("") + "|" + uniq["career_objective"].fillna("") + "|" + uniq["positions"].fillna("")))}
    r["rid"] = key.map(kmap)
    uniq["narr"] = (uniq["career_objective"].fillna("") + ". " + uniq["pos_list"].map(lambda l: ". ".join(l)) + ". " + uniq["cert"] + ". " +
                    uniq["rel_list"].map(lambda l: ", ".join(l)))
    uniq["text"] = uniq["skill_items"].map(lambda l: ", ".join(l)) + ". " + uniq["narr"]

    def label(row):
        for p in row["pos_list"]:
            k = title_to_role(p)
            if k >= 0:
                return k
        return title_to_role(row["career_objective"]) if isinstance(row["career_objective"], str) else -1
    uniq["role"] = uniq.apply(label, axis=1)
    return r, uniq
