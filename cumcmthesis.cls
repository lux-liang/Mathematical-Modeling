\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{cumcmthesis}
              [2015/08/22 v2.0 Standard LaTeX document class]
\newif\if@mcm@bwprint\@mcm@bwprintfalse
\newif\if@mcm@preface\@mcm@prefacetrue
\newcommand\mcm@tokens@keywords{}
\newcommand*\mcm@tokens@tihao{}
\newcommand*\mcm@tokens@baominghao{}
\newcommand*\mcm@tokens@schoolname{}
\newcommand*\mcm@tokens@membera{}
\newcommand*\mcm@tokens@memberb{}
\newcommand*\mcm@tokens@memberc{}
\newcommand*\mcm@tokens@supervisor{}
\newcommand*\mcm@tokens@yearinput{}
\newcommand*\mcm@tokens@monthinput{}
\newcommand*\mcm@tokens@dayinput{}
\DeclareOption{colorprint}{\@mcm@bwprintfalse}
\DeclareOption{bwprint}{\@mcm@bwprinttrue}
\DeclareOption{withoutpreface}{\@mcm@prefacefalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{colorprint}
\ProcessOptions\relax
\LoadClass[a4paper,12pt]{article}
\RequirePackage{ifxetex}
\RequireXeTeX
\ifxetex\else
\ClassError{mcmthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi
\RequirePackage{ctex}
\RequirePackage{geometry}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{longtable}
\RequirePackage{booktabs}
\RequirePackage{indentfirst}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{ulem}
\RequirePackage{calc}
\RequirePackage{hyperref}
\geometry{top=25mm,bottom=25mm,left=25mm,right=25mm}
\renewcommand*{\baselinestretch}{1.38}
\let\mcm@oldtabular\tabular
\let\mcm@endoldtabular\endtabular
\renewenvironment{tabular}%
  {\bgroup%
  \renewcommand{\arraystretch}{1.38}%
  \mcm@oldtabular}%
  {\mcm@endoldtabular\egroup}
\setlength\parindent{2em}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Latin Modern Roman}}
\IfFontExistsTF{Courier New}{\setmonofont{Courier New}}{\setmonofont{Latin Modern Mono}}
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{\setsansfont{Latin Modern Sans}}
\renewcommand\normalsize{%
  \@setfontsize\normalsize{12.05}{14.45}%
  \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
  \abovedisplayshortskip \z@ \@plus3\p@
  \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@}
\newtheorem{definition}{\mcm@cap@definition}
\newtheorem{theorem}{\mcm@cap@theorem}
\newtheorem{lemma}{\mcm@cap@lemma}
\newtheorem{corollary}{\mcm@cap@corollary}
\newtheorem{assumption}{\mcm@cap@assumption}
\newtheorem{conjecture}{\mcm@cap@conjecture}
\newtheorem{axiom}{\mcm@cap@axiom}
\newtheorem{principle}{\mcm@cap@principle}
\newtheorem{problem}{\mcm@cap@problem}
\newtheorem{example}{\mcm@cap@example}
\newtheorem{proof}{\mcm@cap@proof}
\newtheorem{solution}{\mcm@cap@solution}
\renewcommand*{\textfraction}{0.05}
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}
\renewcommand*{\floatpagefraction}{0.85}
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
\graphicspath{{figures/}{figure/}{pictures/}{picture/}{pic/}{pics/}{image/}{images/}}
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\captionsetup[figure]{format=hang,labelsep=quad,font={song,minusfour,bf},position=bottom}
\captionsetup[table]{format=hang,labelsep=quad,font={song,minusfour,bf},position=top}
\setlist{topsep=0.3em,partopsep=0pt,itemsep=0ex plus 0.1ex,parsep=0pt,leftmargin=1.5em,rightmargin=0em,labelsep=0.5em,labelwidth=2em}
\AtBeginDocument{%
  \hypersetup{%
    hyperfootnotes=true,
    hyperindex=true,
    colorlinks=true,
    bookmarksnumbered=true,
    bookmarksopen=true,
    bookmarksopenlevel=0,
    breaklinks=true}%
}
\if@mcm@bwprint
\AtBeginDocument{\hypersetup{hidelinks}}
\else\relax\fi
\renewcommand{\maketitle}{\par
  \begingroup
    \newpage
    \global\@topnum\z@
    \@maketitle
  \endgroup
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \if@mcm@preface
  \null
  \vskip2ex
  \thispagestyle{empty}
  {\zihao{4}\mcm@commit@string@headname\\[-3pt]
  \noindent\rule{\textwidth-1.34em}{1pt}\par}
  \begin{center}%
    {\zihao{-3}\bfseries\heiti \mcm@commit@string@titlea \par}%
    {\vskip1ex\zihao{3}\bfseries\songti \mcm@commit@string@titleb \par}
  \end{center}
    {\zihao{-4} \mcm@commit@string@contents \par}
    {\vskip1ex\zihao{-4}
    \renewcommand{\ULthickness}{0.4pt}\setlength{\ULdepth}{2pt}
    \hspace*{2em}\mcm@commit@string@problemnum\uline{\hspace{1em}\mcm@tokens@tihao\hfill}\makebox[0.66em]{}\par
    \mcm@commit@string@signupnum\uline{\hspace{1em}\mcm@tokens@baominghao\hfill}\makebox[0.66em]{}\par
    \mcm@commit@string@schoolname\uline{\hfill\mcm@tokens@schoolname\hfill}\makebox[0.66em]{}\par
    \newlength{\mcm@lenB}
    \settowidth{\mcm@lenB}{\mcm@commit@string@membername\hspace{1em}1.}
    \setlength{\mcm@lenB}{\textwidth-\mcm@lenB}
    \mcm@commit@string@membername
    \begin{minipage}[t]{\mcm@lenB}
    1.\uline{\hspace{1em}\mcm@tokens@membera\hfill} \makebox[0.46em]{}\par
    2.\uline{\hspace{1em}\mcm@tokens@memberb\hfill} \makebox[0.46em]{}\par
    3.\uline{\hspace{1em}\mcm@tokens@memberc\hfill} \makebox[0.46em]{}\par
   \end{minipage}\par\vskip1ex
    \mcm@commit@string@supervisorname\uline{\hspace{1em}\mcm@tokens@supervisor\hfill}\makebox[0.66em]{}
    \vskip2ex
    \newlength{\mcm@lenA}
    \settowidth{\mcm@lenA}{请仔细核对，提交后将不再允许做任何修改。如}
    \hspace*{\mcm@lenA}\mcm@commit@string@date\hspace{0.5em}\uline{\hfill\hspace{1em}\mcm@tokens@yearinput\hspace{1em}\hfill}%
    \mcm@commit@string@year\uline{\hfill\mcm@tokens@monthinput\hfill}\mcm@commit@string@month%
    \uline{\hfill\mcm@tokens@dayinput\hfill}\mcm@commit@string@day\makebox[1em]{}\par}
    \vskip1ex
    {\kaishu\bfseries\mcm@commit@string@inform\par}
    \vfill
    \null
  \fi
  \makeothertitle
}
\def\makeothertitle{%
  \if@mcm@preface
  \par
  \newpage
  \null
  \thispagestyle{empty}
  {\zihao{4}\mcm@numberpage@string@headname\\[-5pt]
  \noindent\hspace*{0.67em}\rule{\textwidth-1.33em}{1pt}\par}
  \begin{center}
  {\zihao{4}\bfseries\heiti \mcm@numberpage@string@titlea\par}
  {\vskip1ex\zihao{3}\bfseries\songti \mcm@numberpage@string@titleb\par}
  \end{center}\par
  {%
  \zihao{4}
  \vskip3em
  \hspace*{2em}\mcm@numberpage@string@appraiselog
  \par
  \newlength{\mcm@lenC}
  \setlength{\mcm@lenC}{\textwidth-8.72em}
  \setlength{\mcm@lenC}{\mcm@lenC/6}
  \newlength{\mcm@lenD}
  \settoheight{\mcm@lenD}{\mcm@numberpage@string@reviewer}
  \setlength{\mcm@lenD}{\mcm@lenD+2em}
  \newlength{\mcm@lenF}
  \settoheight{\mcm@lenF}{\mcm@numberpage@string@remarks}
  \setlength{\mcm@lenF}{\mcm@lenF+2em}
  \newcolumntype{P}{@{}p{\mcm@lenC}@{}}
  \hspace*{1.5em}\begin{tabular}{|!{\hspace{4pt}}c!{\hspace{4pt}}|*{6}{P|}}
  \hline
  \rule{0pt}{\mcm@lenD}\raisebox{0.6em}[0pt][0pt]{\mcm@numberpage@string@reviewer} & & & & & &  \\\cline{1-7}
  \rule{0pt}{\mcm@lenF}\raisebox{0.6em}[0pt][0pt]{\mcm@numberpage@string@remarks} & & & & & &  \\
  \hline
  \end{tabular}\par
  \vskip5em
  \hspace*{2em}\mcm@numberpage@string@unifiednum
  \vskip7em
  \hspace*{2em}\mcm@numberpage@string@countrynum}
  \vskip7em
  {\zihao{-4}\kaishu\bfseries\mcm@numberpage@string@inform\par}
  \vfill
  \null
  \newpage
  \null
  \setcounter{page}{1}
  {\centering \zihao{3}\bfseries \@title\par}
  \vskip1ex
  \else
  \setcounter{page}{1}
  {\centering \zihao{3}\bfseries \@title\par}
  \vskip1ex
  \fi
}
\renewcommand\contentsname{\mcm@cap@contentsname}
\renewcommand\listfigurename{\mcm@cap@listfigurename}
\renewcommand\listtablename{\mcm@cap@listtablename}
\renewcommand\refname{\mcm@cap@refname}
\renewcommand\indexname{\mcm@cap@indexname}
\renewcommand\figurename{\mcm@cap@figurename}
\renewcommand\tablename{\mcm@cap@tablename}
\renewcommand\appendixname{\mcm@cap@appendixname}
\renewcommand\abstractname{\mcm@cap@abstractname}
\setcounter{secnumdepth}{4}
\renewcommand\thesection{\chinese{section}、\kern-1em}
\renewcommand\thesubsection{\arabic{section}\thinspace.\thinspace\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection\thinspace.\thinspace\arabic{subsubsection}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\centering\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
  {3.25ex \@plus1ex \@minus.2ex}%
  {-1em}%
  {\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
   {3.25ex \@plus1ex \@minus .2ex}%
   {-1em}%
   {\normalfont\normalsize\bfseries}}
\renewenvironment{abstract}{%
\if@twocolumn
    \section*{\abstractname}%
  \else
   \begin{center}%
  {\zihao{4}\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
   \end{center}%
   \quotation
  \fi}
  {\if@twocolumn\else\endquotation\newpage\null\fi}
\renewenvironment{quotation}
  {\list{}{\listparindent 2em%
   \itemindent \listparindent
   \rightmargin\z@
   \leftmargin\z@
   \parsep \z@ \@plus\p@}%
  \item\relax}
  {\endlist}
\newcommand\keywords[1]{%
  \renewcommand{\mcm@tokens@keywords}{#1}
  \par
  \vskip1ex
  {\noindent\zihao{-4}\heiti\mcm@cap@keywordsname：}~{\mcm@tokens@keywords}
}
\renewcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\appendixname\@Alph\c@section}}
\renewenvironment{thebibliography}[1]
     {\section{\refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand*\tihao[1]{\renewcommand{\mcm@tokens@tihao}{#1}}
\newcommand*\baominghao[1]{\renewcommand{\mcm@tokens@baominghao}{#1}}
\newcommand*\schoolname[1]{\renewcommand{\mcm@tokens@schoolname}{#1}}
\newcommand*\membera[1]{\renewcommand{\mcm@tokens@membera}{#1}}
\newcommand*\memberb[1]{\renewcommand{\mcm@tokens@memberb}{#1}}
\newcommand*\memberc[1]{\renewcommand{\mcm@tokens@memberc}{#1}}
\newcommand*\supervisor[1]{\renewcommand{\mcm@tokens@supervisor}{#1}}
\newcommand*\yearinput[1]{\renewcommand{\mcm@tokens@yearinput}{#1}}
\newcommand*\monthinput[1]{\renewcommand{\mcm@tokens@monthinput}{#1}}
\newcommand*\dayinput[1]{\renewcommand{\mcm@tokens@dayinput}{#1}}
\newcommand*{\mcm@cap@definition}{定义}
\newcommand*{\mcm@cap@theorem}{定理}
\newcommand*{\mcm@cap@lemma}{引理}
\newcommand*{\mcm@cap@corollary}{推论}
\newcommand*{\mcm@cap@assumption}{假设}
\newcommand*{\mcm@cap@conjecture}{猜想}
\newcommand*{\mcm@cap@axiom}{公理}
\newcommand*{\mcm@cap@principle}{定律}
\newcommand*{\mcm@cap@problem}{问题}
\newcommand*{\mcm@cap@example}{例}
\newcommand*{\mcm@cap@proof}{证明}
\newcommand*{\mcm@cap@solution}{解}
\newcommand*{\mcm@cap@contentsname}{目录}
\newcommand*{\mcm@cap@listfigurename}{插图清单}
\newcommand*{\mcm@cap@listtablename}{附表清单}
\newcommand*{\mcm@cap@refname}{参考文献}
\newcommand*{\mcm@cap@indexname}{索引}
\newcommand*{\mcm@cap@figurename}{图}
\newcommand*{\mcm@cap@tablename}{表}
\newcommand*{\mcm@cap@appendixname}{附录}
\newcommand*{\mcm@cap@abstractname}{摘要}
\newcommand*{\mcm@cap@keywordsname}{关键字}
\newcommand*\mcm@commit@string@headname{赛区评阅编号（由赛区组委会填写）：}
\newcommand*\mcm@commit@string@titlea{\the\year 高教社杯全国大学生数学建模竞赛}
\newcommand*\mcm@commit@string@titleb{承\hspace{1em}诺\hspace{1em}书}
\newcommand\mcm@commit@string@contents{\quotation
 我们仔细阅读了《全国大学生数学建模竞赛章程》和《全国大学生数学建模竞赛参赛规则》%
\hspace*{-0.6em}（以下简称为“竞赛章程和参赛规则”，\hspace{-0.2em}可从全国大学生数学建模竞赛网站下载）。

我们完全明白，在竞赛开始后参赛队员不能以任何方式（包括电话、电子邮件、网上咨询等）%
与队外的任何人（包括指导教师）研究、讨论与赛题有关的问题。

 我们知道，抄袭别人的成果是违反竞赛章程和参赛规则的，如果引用别人的成果或其他公开的资料%
（包括网上查到的资料），必须按照规定的参考文献的表述方式在正文引用处和参考文献中明确列出。

我们郑重承诺，严格遵守竞赛章程和参赛规则，以保证竞赛的公正、公平性。%
如有违反竞赛章程和参赛规则的行为，我们将受到严肃处理。

 我们授权全国大学生数学建模竞赛组委会，可将我们的论文以任何形式进行公开展示%
（包括进行网上公示，在书籍、期刊和其他媒体进行正式或非正式发表等）。
\endquotation
}
\newcommand*\mcm@commit@string@problemnum{我们参赛选择的题号（从A/B/C/D中选择一项填写）：}
\newcommand*\mcm@commit@string@signupnum{我们的报名参赛队号（12位数字全国统一编号）：}
\newcommand*\mcm@commit@string@schoolname{参赛学校（完整的学校全称，不含院系名）：}
\newcommand*\mcm@commit@string@membername{参赛队员 (打印并签名) ：}
\newcommand*\mcm@commit@string@supervisorname{指导教师或指导教师组负责人  (打印并签名)：}
\newcommand\mcm@commit@string@inform{\quotation
\noindent（此承诺书打印签名后作为纸质论文的封面，注意电子版论文中不得出现此页。%
以上内容请仔细核对，如填写错误，论文可能被取消评奖资格。）
\endquotation
}
\newcommand*\mcm@commit@string@date{日期:}
\newcommand*\mcm@commit@string@year{年}
\newcommand*\mcm@commit@string@month{月}
\newcommand*\mcm@commit@string@day{日}
\newcommand*\mcm@commit@string@appraisenum{赛区评阅编号（由赛区组委会评阅前进行编号）：}
\newcommand*\mcm@numberpage@string@headname{赛区评阅编号（由赛区组委会填写）：}
\newcommand*\mcm@numberpage@string@titlea{\the\year 高教社杯全国大学生数学建模竞赛}
\newcommand*\mcm@numberpage@string@titleb{编\hspace{.5em}号\hspace{.5em}专\hspace{.5em}用\hspace{.5em}页}
\newcommand*\mcm@numberpage@string@appraiselog{赛区评阅记录（可供赛区评阅时使用）：}
\newcommand\mcm@numberpage@string@reviewer{\shortstack{评\\ 阅\\ 人}}
\newcommand\mcm@numberpage@string@remarks{\shortstack{备\\ 注}}
\newcommand*\mcm@numberpage@string@unifiednum{送全国评阅统一编号（由赛区组委会填写）：}
\newcommand*\mcm@numberpage@string@countrynum{全国评阅随机编号（由全国组委会填写）：}
\newcommand\mcm@numberpage@string@inform{\quotation
\noindent（此编号专用页仅供赛区和全国评阅使用，参赛队打印后装订到纸质论文的第二页上。%
注意电子版论文中不得出现此页，即电子版论文的第一页为标题、摘要和关键词页。）
\endquotation
}
\newcommand{\upcite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}
\newcommand{\citing}[1]{\textsuperscript{\cite{#1}}}
\setlist{noitemsep}
\endinput
