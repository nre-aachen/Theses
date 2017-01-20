%--------------------------------------------------------------------------------------
%        IDEA League Joint Master's in Applied Geophysics Standard MSc Thesis Lay-out
%   
%				 	   N. Grobbe, MSc (based on framework F.C. Schoemaker)
%--------------------------------------------------------------------------------------
%%
%% Thesis style developed by
%% 
%% Niels Grobbe
%% Delft University of Technology
%% Faculty of Civil Engineering and Geosciences
%% Department Geoscience & Engineering
%% Section of Applied Geophysics and Petrophysics
%% Delft, the Netherlands, November 2013
%% based on LaTeX `book.cls'
%%-------------------------------------------------------------------------------------
%% This specific file is the class file, which is needs to be specified in the command \documentclass{}
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{mscThesis}
              [2013/11/20 v1.0f
 Standard IDEA League Applied Geophysics MSc thesis class]
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%    MSc Thesis Definitions
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newcommand*{\mscName}[1]%
            {\def\mscname{#1}}
\newcommand*{\mscTitle}[1]%
            {\def\msctitle{#1}}
\newcommand*{\mscSubTitle}[1]%
            {\def\mscsubtitle{#1}}
\newcommand*{\mscDate}[1]%
            {\def\mscdate{#1}}
\newcommand*{\mscBackPicture}[1]%
            {\def\mscbackpicture{#1}}
\newcommand*{\mscReaderOne}[1]%
            {\def\mscreaderone{#1}}
\newcommand*{\mscReaderTwo}[1]%
            {\def\mscreadertwo{#1}}
\newcommand*{\mscReaderThree}[1]%
            {\def\mscreaderthree{#1}}
\newcommand*{\mscReaderFour}[1]%
            {\def\mscreaderfour{#1}}
\newcommand*{\mscKeyWords}[1]%
            {\def\msckeywords{#1}}
\mscName{}
\mscTitle{}
\mscSubTitle{}
\mscKeyWords{}
\mscDate{\today}
\mscBackPicture{}
\mscReaderOne{}
\mscReaderTwo{}
\mscReaderThree{}
\mscReaderFour{}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%    PAPER SIZE DEFINITIONS: cf STANDARD BOOK CLASS
%%		Here you can adjust your paper size if desired
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility
\@openrighttrue
\else
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\ExecuteOptions{a4paper,11pt,twoside,onecolumn,final,openright}
\ProcessOptions
\input{bk1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
      \def\chaptermark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
  \if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going to the top of the page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
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
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@arabic\c@chapter}
\renewcommand \thesection {\thechapter-\@arabic\c@section}
\renewcommand\thesubsection   {\thesection-\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection-\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection-\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph-\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand\frontmatter{%
    \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{roman}}
\newcommand\mainmatter{%
    \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand\part{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \null\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \sffamily
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries \partname\nobreakspace\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \sffamily
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
               \if@openright
                \null
                \thispagestyle{empty}%
                \newpage
               \fi
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \huge\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\sffamily\Large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\sffamily\large\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\sffamily\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\sffamily\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\sffamily\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Commands
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Environments
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
% --- Define Annexdote Style ---
\RequirePackage{ntheorem}
\newcounter{Anecdote}
\setcounter{Anecdote}{1} \theoremstyle{plain}
%\theoremheaderfont{\slshape}
\theoremheaderfont{\scshape} \theorembodyfont{\upshape}
\theoremindent1.5cm
\newtheorem{anecdote}{Anecdote}[Anecdote]
%\renewcommand{\theanecdote}{\thechapter.\arabic{anecdote}}
\renewcommand{\theanecdote}{\arabic{anecdote}}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  TitlePages
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\if@compatibility
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Part, Chapter, Section, Subsection, ...
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand\tableofcontents{%
    \thispagestyle{empty}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           \contentsname}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    \cleardoublepage
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries \sffamily #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries \sffamily
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@section{\sffamily \@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\sffamily \@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\sffamily \@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\sffamily \@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\sffamily \@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename}%
      \@mkboth{\listfigurename}%
              {\listfigurename}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@figure{\sffamily \@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename}%
      \@mkboth{%
          \listtablename}%
         {\listtablename}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname}%
      \@mkboth{\bibname}{\bibname}%
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
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\indexname}%
                        {\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 1cm 
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
%\newcommand{\hyperpage}[1]{\dotfill\hyperpage{#1}}
\newcommand\@idxitem{\par\hangindent 40\p@ }
\newcommand\subitem{\@idxitem \hspace*{20\p@} }
\newcommand\subsubitem{\@idxitem \hspace*{30\p@} }
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Table of Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
% \if@twoside
% \else
% \raggedbottom
% \fi
% \if@twocolumn
%   \twocolumn
%   \sloppy
%   \flushbottom
% \else
  \onecolumn
% \fi
\usepackage{rotating} %package to rotate e.g. Matrices from portrait to landscape mode.
\usepackage{pifont}
\usepackage{syntonly} %added in combination with \syntaxonly command
%\syntaxonly  %if set 'on', only scans for correct syntax, no DVI file created: saves time
\usepackage{makeidx} %added in order to make index correctly
\usepackage{lscape} %package that turns specified part into landscape orientation
%--------------------------------------------------------------------------------------
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  HyperReference: moved from here to the end of this class file (see 'the not so short intro...' p. 93)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
%\usepackage[pdftex]{thumbpdf}
%\RequirePackage[bookmarks=true,bookmarksnumbered=true,dvips,pdftex,hypertexnames=false,breaklinks=true,ps2pdf]{hyperref}
%\RequirePackage[bookmarks=true,hypertexnames=false,breaklinks=true,bookmarksnumbered=true,dvips,ps2pdf]{hyperref}
             %cancelled out because warning during compiling: command pagecolor not available anymore?!
%%            pdfborder=[0 0 0],}
%%
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Extra Packages
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\RequirePackage{a4wide}
\RequirePackage{fancyhdr}
\RequirePackage[leftbars]{changebar}%{Stylestuff/changebar}
\RequirePackage{theorem}
%\RequirePackage[normal]{caption2}
%\RequirePackage[hang]{caption2}
\RequirePackage[hang]{caption}
\RequirePackage{graphicx}
\RequirePackage{psfrag}
\RequirePackage{nomencl} %{StyleStuff/nomencl}
\RequirePackage{makeidx}
\RequirePackage{color,ifthen,epsfig}
\RequirePackage{eso-pic} %{StyleStuff/eso-pic}
\RequirePackage{fncychap}%{StyleStuff/fncychap}
\RequirePackage{sidecap}
\RequirePackage{amsmath,amssymb,nicefrac,eurosym}
%%%\RequirePackage{latexcad}%{StyleStuff/latexcad}   % latexcad.sty for drawings etc --> ADD TeXnicCentre: gives error in TeXMaker!!
    \setlength{\unitlength}{1mm}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Matlab and C++ Listings
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
%
%---MATLAB INSERTING---%
\RequirePackage{listings}
\RequirePackage{colortbl}
%
\lstloadlanguages{Matlab}
%
\lstset{language=Matlab, numbers = right, stepnumber = 5,breaklines=true}
%
\definecolor{MATdarkgreen}{rgb}{0.1333,0.545,0.1333}
\definecolor{MATdarkred}{rgb}{0.698,0,0}
%
\lstset{basicstyle=\tiny, tabsize=4,
        stringstyle=\ttfamily\color{MATdarkred},
        keywordstyle=\ttfamily\color{blue},
        identifierstyle=\ttfamily,
        showstringspaces=false,
        commentstyle=\ttfamily\color{MATdarkgreen}}
% EXTRA: \bfseries \underbar \ttfamily

\newcommand{\matlab}{\textsc{Matlab}}

%
% --- Use Acronyms ---
\RequirePackage{acronym}
\renewcommand{\ac}[1]{%
  \expandafter\ifx\csname ac@#1\endcsname\relax%
    \acf{#1}%
    \expandafter\gdef\csname ac@#1\endcsname{x}%
  \else%
    \acs{#1}%
  \fi}
%
%--------------------------------------------------------------------------------------
\renewcommand{\captionfont}{\small\sffamily}
\renewcommand{\captionlabelfont}{\small\bfseries\sffamily}
\setlength{\captionmargin}{20pt}
%For fncychap
\makeatletter
  %\ChNameUpperCase
  %\ChTitleUpperCase
  \ChNameVar{\raggedleft\Huge\rm}
  \ChNumVar{\Huge\rm}
  \ChTitleVar{\raggedleft\Huge\sf\bfseries}
  \ChRuleWidth{2pt}

  \renewcommand{\DOCH}{%
    %\mghrulefill{3\RW}\par\nobreak % vette lijn
    \vskip -0.5\baselineskip
    \mghrulefill{\RW}\par\nobreak
    \CNV\FmN{\@chapapp}\space \CNoV\thechapter
    \par\nobreak
    \vskip -0.5\baselineskip
   }
  \renewcommand{\DOTI}[1]{%
    \mghrulefill{\RW}\par\nobreak
    \CTV\FmTi{#1}\par\nobreak
    \vskip 60\p@
    }
  \renewcommand{\DOTIS}[1]{%
    \mghrulefill{\RW}\par\nobreak
    \CTV\FmTi{#1}\par\nobreak
    \vskip 60\p@
    }
\makeatother

\definecolor{dark}{gray}{.25}

%REQUIRES "FANCYHDR.STY"
%-------------------------------------------------------------------------------------------
%                            Header definitions... (Clark Borst)
%-------------------------------------------------------------------------------------------
\pagestyle{fancyplain}
\addtolength{\headheight}{\baselineskip} %add some space to headheight to eliminate LaTeX "Overfull \vbox..." warnings
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}
%\fancyhead{}
%\newcommand{\lfig}{\begin{picture}(0,0)
%\put(-50,-5){\includegraphics[width=2.0cm]{TAlogo}}
%\end{picture}}
%\newcommand{\rfig}{\begin{picture}(0,0)
%\put(0,-5){\includegraphics[width=2.0cm]{TAlogo}}
%\end{picture}}
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries\rightmark}}
\rhead[\fancyplain{}{\bfseries\leftmark}]{\fancyplain{}{\bfseries\thepage}}
\cfoot{}

%--------------------------------------------------------------------------------------

%REQUIRES PACKAGE "THEOREM.STY" AND "CHANGEBAR.STY"
%------------------------------------------------------------------------------------%
%                       EXAMPLE ENVIRONMENT (Clark Borst)                            %
%------------------------------------------------------------------------------------%
\setlength{\changebarwidth}{5mm} \setlength{\changebarsep}{-20pt}
\setcounter{changebargrey}{75}
\theoremstyle{break} \theorembodyfont{\normalfont\small}
\theoremheaderfont{\sffamily\bfseries\Large}
\newtheorem{example}{EXAMPLE}[chapter]
\newenvironment{ex}{\begin{quote}%
  \begin{example} \begin{changebar}}{\end{changebar} \end{example} \end{quote}}%
\newenvironment{exa}{\vspace*{\baselineskip} \begin{ex}}%
                    {\end{ex} \vspace*{\baselineskip}}
%------------------------------------------------------------------------------------%
%                            EXAMPLE ENVIRONMENT                                     %
%------------------------------------------------------------------------------------%
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}

\renewcommand{\Re}{\mathfrak{Re}}
\renewcommand{\Im}{\mathfrak{Im}}

\setlength{\parskip}{\medskipamount}
\setlength{\parindent}{0pt}

\setcounter{tocdepth}{3}

%\renewcommand{\cfttoctitlefont}%
%{   \vskip -0.5\baselineskip
%    \mghrulefill{\RW}\par\nobreak \vskip 0.5\baselineskip
%    \raggedleft
%     \Huge\bfseries\sffamily}
%\renewcommand{\cftchapfont}{\large\bfseries\sffamily}
%\renewcommand{\cftsecfont}{\sffamily}
%\renewcommand{\cftsubsecfont}{\sffamily}
%\renewcommand{\cftsubsubsecfont}{\sffamily}
%\renewcommand{\cftpartfont}{\Large\bfseries\sffamily}
%\renewcommand{\cftmarktoc}{\markboth{\bfseries\contentsname}{\bfseries\contentsname}}
%\setlength{\cftaftertoctitleskip}{2cm}
%
%\renewcommand{\cftloftitlefont}%
%{\Huge\bfseries\sffamily}
%\setlength{\cftafterloftitleskip}{2cm}
%\renewcommand{\cftfigfont}{\sffamily}
%\renewcommand{\cftmarklof}{\markboth{\bfseries\listfigurename}{\bfseries\listfigurename}}
%
%\renewcommand{\cftlottitlefont}%
%{\Huge\bfseries\sffamily}
%\setlength{\cftafterlottitleskip}{2cm}
%\renewcommand{\cfttabfont}{\sffamily}
%\renewcommand{\cftmarklot}{\markboth{\bfseries\listtablename}{\bfseries\listtablename}}
%--------------------------------------------------------------------------------------
\renewcommand{\nomname}{Nomenclature}
\setlength{\nomitemsep}{-0.5\parsep}

\newcommand{\abbrev}[2]
   {#2 (#1)\nomenclature[a ]{#1}{\MakeUppercase#2}}

\newcommand{\lsymb}[2]
   {#1\nomenclature[c ]{#1}{\MakeUppercase#2}}

\newcommand{\gsymb}[2]
   {#1\nomenclature[b ]{#1}{\MakeUppercase#2}}

\newcommand{\subs}[2]
   {\nomenclature[d ]{$ _{#1} $}{\MakeUppercase#2}}

\newcommand{\supers}[2]
   {\nomenclature[e ]{$ ^{#1} $}{\MakeUppercase#2}}

\newcommand{\others}[2]
   {\nomenclature[f ]{#1}{\MakeUppercase#2}}


\renewcommand{\nomgroup}[1]{%
  \ifthenelse{\equal{#1}{A}}{%
  \medskip\item[\LARGE\textbf{\textsf{Abbreviations}}]\medskip}{%
  \ifthenelse{\equal{#1}{C}}{%
   \medskip\item[\LARGE\textbf{\textsf{Latin Symbols}}]\medskip}{%
  \ifthenelse{\equal{#1}{B}}{%
   \medskip\item[\LARGE\textbf{\textsf{Greek Symbols}}]\medskip}{%
  \ifthenelse{\equal{#1}{D}}{%
   \medskip\item[\LARGE\textbf{\textsf{Subscripts}}]\medskip}{%
  \ifthenelse{\equal{#1}{E}}{%
  \medskip\item[\LARGE\textbf{\textsf{Superscripts}}]\medskip}{%
    \ifthenelse{\equal{#1}{F}}{%
   \medskip\item[\LARGE\textbf{\textsf{Other}}]\medskip}{}%
   }}}}}}
%--------------------------------------------------------------------------------------
\makeatletter
\newcommand\BackgroundPicture[1]{%
    \setlength{\unitlength}{1pt}%
    \put(0,\strip@pt\paperheight){%
    \parbox[t][\paperheight]{\paperwidth}{%
    \vfill
    \centering\includegraphics{#1}
    \vfill
}}}
\newcommand\TuPicture[1]{%
    \setlength{\unitlength}{1pt}%
    \put(0,\strip@pt\paperheight){%
    \parbox[t][\paperheight]{\paperwidth}{%
    \vfill
    \centering\includegraphics{#1}
}}}
\makeatother
%--------------------------------------------------------------------------------------
\newcommand{\eqnref}[1]{Eq.\ (\ref{#1})}
\newcommand{\figref}[1]{Figure~\ref{#1}}
\newcommand{\tabref}[1]{Table~\ref{#1}}
%--------------------------------------------------------------------------------------
\renewcommand{\maketitle}{%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main title with picture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{titlepage}
\ifthenelse{ \equal{\mscbackpicture}{} }{}{
\AddToShipoutPicture*{\BackgroundPicture{\mscbackpicture}}
}
%\AddToShipoutPicture*{\TuPicture{StyleStuff/TUD_banner_A4_ENG_edited}}
    \begin{center}
     \includegraphics[width=8cm]{IDEA_League_Logo}
    \end{center} 
 \begin{center}
% \huge{\textsc{IDEA League}}\\
 \vspace{1em}
 \LARGE{\textsc{Master of Science in Applied Geophysics Thesis}} \\
 \end{center}%
 \newcommand{\HRule}{\rule{\linewidth}{1mm}}
 \setlength{\parindent}{0mm}
 \setlength{\parskip}{0mm}
 \vspace*{\stretch{2}}
 \HRule
 \begin{flushright}
 \Huge{\textbf{\textsf{\msctitle}}}\\ \Large{\textbf{\textsf{\mscsubtitle}}} \\[5mm]
 \Large{\textbf{\textsf{ \mscname}}}
 \end{flushright}
 \HRule
 \\ \mscdate \\
 \vspace*{\stretch{2}}
 \hspace{1cm}
\end{titlepage}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Second title page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\clearemptydoublepage
\thispagestyle{empty}
\setcounter{page}{1}
\begin{center}
\Huge{\textbf{\textsf{\msctitle}}}\\ \Large{\textbf{\textsf{\mscsubtitle}}} \\[1cm]
\Large{\textsc{Master of Science Thesis}}\\[2cm]
\Large{for the degree of Master of Science in Applied Geophysics at \par 
Delft University of Technology \par % pick here the university where you carried out your MSc research
ETH Z\"{u}rich \par
RWTH Aachen University\par
by}\\[3cm]
\Large{\mscname}\\[1cm]
\Large{\mscdate}
\end{center}
 \vspace*{\stretch{2}}
 %\begin{center} \epsfig{file=cslogo.eps,width=3cm} \end{center}
 %
 \begin{center}
 \begin{tabular}{lcr}
 Department of Geoscience \& Engineering & $\cdot$ & Delft University of Technology \\ % Change here for different universities and their corresponding departments.
  \end{tabular}
 \end{center}


%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\vspace*{13cm}
\vspace*{\stretch{3}}
\thispagestyle{empty}
\begin{figure}[h!] % PUT HERE the link and logo of the university where you carried out your MSc thesis %research
  \href{http://www.tudelft.nl}{\includegraphics{StyleStuff/tud}}
  \label{fig:TUDlogo}
\end{figure}
Copyright $\copyright$ 2013 by IDEA League Joint Master's in Applied Geophysics:
\par
Delft University of Technology, ETH Z\"{u}rich, RWTH Aachen University % pick here the university where %you carried out your MSc research
\par
All rights reserved.\\
No part of the material protected by this copyright notice may be reproduced or utilized in any form or by any means, electronic or mechanical, including photocopying or by any information storage and retrieval system, without permission from this publisher.
\par
Printed in The Netherlands, Switzerland, Germany % pick here the country of the university where you carried out your MSc research
\cleardoublepage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Readers Page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\thispagestyle{empty}
\vspace{10em}
\begin{center}
     \large IDEA LEAGUE \\
     JOINT MASTER'S IN APPLIED GEOPHYSICS \\
     \vspace{5em}
    	 Delft University of Technology, The Netherlands\\
    	 ETH Z\"{u}rich, Switzerland  \\
    	 RWTH Aachen, Germany
\end{center}

\vspace{1cm}

\vspace{4cm}
%\vspace{4cm}
%\vspace{4cm}
%\vspace{6cm}

\mbox{} \hfill Dated: $\textit{ \today }$
\vspace{3cm}

\noindent
Supervisor(s):    \hfill \underline{ \hspace{7cm} }\\
\mbox{}        \hfill  \mscreaderone \\
\\
\mbox{}        \hfill \underline{ \hspace{7cm} }\\
\mbox{}        \hfill  \mscreadertwo\\
%\hfill \underline{ \hspace{7cm} }\\
%\mbox{}        \hfill  \mscreaderthree\\
\\
\\
Committee Members:       \hfill \underline{ \hspace{7cm} }\\
\mbox{}        \hfill  \mscreaderone\\
\ifthenelse{ \equal{\mscreadertwo}{} }{}{\\
\\
\mbox{}        \hfill \underline{ \hspace{7cm} }\\
\mbox{}        \hfill \mscreadertwo\\
}
\ifthenelse{ \equal{\mscreaderthree}{} }{}{\\
\\
\mbox{}        \hfill \underline{ \hspace{7cm} }\\
\mbox{}        \hfill \mscreaderthree\\
}
%%\ifthenelse{ \equal{\mscreaderfour}{} }{}{\\
%%\\
%%\mbox{}        \hfill \underline{ \hspace{7cm} }\\
%%\mbox{}        \hfill \mscreaderfour \\
%}
\newpage\clearemptydoublepage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fill out PDF author - title
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\setThesisInfo}{
\makeglossary
\makeindex
\hypersetup{pdftitle    = MSc Thesis: \msctitle,
            pdfauthor   = \mscname,
            pdfsubject  = \mscsubtitle,
            pdfkeywords = \msckeywords}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TOC / LOF / LOT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\toc}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
}

\newcommand{\toclof}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
    \listoffigures\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
}

\newcommand{\tocloflot}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
    \listoffigures\cleardoublepage
    \listoftables\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
    \cleardoublepage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chapter with no numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\nonumchap}[1]{%
    \chapter*{#1}
    \chaptermark{#1}
    \markboth{#1}{#1}
    \addcontentsline{toc}{chapter}{\numberline{}#1}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print nomenclature list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printnomencl}{
\addcontentsline{toc}{chapter}{\numberline{}\nomname}
\markboth{\nomname}{\nomname}
\printglossary[0.8in]
\cleardoublepage
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print bib list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printbib}[1]{
\cleardoublepage
\addcontentsline{toc}{chapter}{\numberline{}\bibname}
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries \bibname}}
\rhead[\fancyplain{}{\bfseries \bibname}]{\fancyplain{}{\bfseries\thepage}}
\bibliography{#1}
\cleardoublepage
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries\rightmark}}
\rhead[\fancyplain{}{\bfseries\leftmark}]{\fancyplain{}{\bfseries\thepage}}
\cfoot{}
}
%--------------------------------------------------------------------------------------
\lfoot[\scriptsize \textsf{\today}]{\scriptsize \textsf{}}					
\rfoot[\scriptsize \textsf{}]{\scriptsize \textsf{\today}}               
%--------------------------------------------------------------------------------------
\sloppy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  HyperReference: moved from here to the end of this class file 
%%  (see 'The not so short introduction to LaTeX...' p. 93)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[dvips,ps2pdf]{hyperref}
\RequirePackage{colortbl}
%
\definecolor{darkred}{rgb}{0.5,0,0}
\definecolor{darkgreen}{rgb}{0,0.3,0}
\definecolor{darkblue}{rgb}{0,0,0.5}
\definecolor{darkbrown}{rgb}{0.28,0.07,0.07}
%
\hypersetup{colorlinks=true,
            citecolor=darkblue,
            urlcolor=darkblue,
            linkcolor=darkblue,
            menucolor=darkblue,
            anchorcolor=red,}            
\endinput
%%
%% End of file `mscThesis.cls'.



