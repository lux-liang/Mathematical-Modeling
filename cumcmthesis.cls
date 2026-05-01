\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{cumcmthesis}
              [2026/04/28 v3.0 Minimal mathematical modeling paper class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper,12pt]{article}

\RequirePackage{ifxetex}
\RequireXeTeX
\ifxetex\else
\ClassError{cumcmthesis}{You must use xelatex}{Please compile this document with xelatex.}
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
\RequirePackage{hyperref}
\RequirePackage{fontspec}

\geometry{top=25mm,bottom=25mm,left=25mm,right=25mm}
\renewcommand*{\baselinestretch}{1.35}
\setlength\parindent{2em}

\setmainfont{Latin Modern Roman}
\setmonofont{Latin Modern Mono}
\setsansfont{Latin Modern Sans}

\renewcommand\normalsize{%
  \@setfontsize\normalsize{12.05}{14.45}%
  \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
  \abovedisplayshortskip \z@ \@plus3\p@
  \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@}

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
    hidelinks,
    bookmarksnumbered=true,
    bookmarksopen=true,
    bookmarksopenlevel=0,
    breaklinks=true
  }%
}

\def\@contestname{}
\def\@school{}
\def\@teamname{}
\def\@members{}
\def\@advisor{}
\def\@keywords{}

\newcommand{\contestname}[1]{\gdef\@contestname{#1}}
\newcommand{\school}[1]{\gdef\@school{#1}}
\newcommand{\teamname}[1]{\gdef\@teamname{#1}}
\newcommand{\members}[1]{\gdef\@members{#1}}
\newcommand{\advisor}[1]{\gdef\@advisor{#1}}

\newcommand\keywords[1]{%
  \gdef\@keywords{#1}%
  \par\vskip1ex
  {\noindent\zihao{-4}\heiti 关键词：}#1
}

\renewcommand\contentsname{目录}
\renewcommand\listfigurename{插图清单}
\renewcommand\listtablename{附表清单}
\renewcommand\refname{参考文献}
\renewcommand\figurename{图}
\renewcommand\tablename{表}
\renewcommand\appendixname{附录}
\renewcommand\abstractname{摘要}

\setcounter{secnumdepth}{4}
\renewcommand\thesection{\chinese{section}、\kern-1em}
\renewcommand\thesubsection{\arabic{section}.\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}

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

\renewenvironment{abstract}{%
\if@twocolumn
  \section*{\abstractname}%
\else
  \begin{center}%
    {\zihao{4}\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
  \end{center}%
  \quotation
\fi}
{\if@twocolumn\else\endquotation\fi}

\renewenvironment{quotation}
  {\list{}{\listparindent 2em%
   \itemindent \listparindent
   \rightmargin\z@
   \leftmargin\z@
   \parsep \z@ \@plus\p@}%
  \item\relax}
  {\endlist}

\renewcommand{\maketitle}{%
  \begin{center}
    {\zihao{2}\bfseries \@title \par}
    \vskip 0.8em
    {\zihao{4}\@contestname \par}
    \vskip 1.2em
    \begin{tabular}{rl}
      学校名称： & \@school \\
      队伍编号： & \@teamname \\
      参赛队员： & \@members \\
      指导教师： & \@advisor \\
      完成日期： & \@date \\
    \end{tabular}
  \end{center}
  \vskip 1.2em
}

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

\endinput
