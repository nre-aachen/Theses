;; Change this variable to directory where you put the pictures.
(defvar LaTeX-pifont-font-table-directory "~/.xemacs/style")

(TeX-add-style-hook "pifont"
 (function
  (lambda ()
    (LaTeX-add-environments
     '("dingautolist" LaTeX-pifont-env-ding)
     '("dinglist"     LaTeX-pifont-env-ding)
     '("Piautolist"   LaTeX-pifont-env-pi)
     '("Pilist"       LaTeX-pifont-env-pi))
    (TeX-add-symbols
     '("Piline"   (LaTeX-pifont-cmd-pi 2))
     '("Pifill"   (LaTeX-pifont-cmd-pi 2))
     '("Pisymbol" (LaTeX-pifont-cmd-pi 2))
     '("Pifont"   (LaTeX-pifont-cmd-pi 1))
     '("ding"     LaTeX-pifont-cmd-ding)
     '("dingfill" LaTeX-pifont-cmd-ding)
     '("dingline" LaTeX-pifont-cmd-ding)
     ))))

(defvar LaTeX-pifont-font-alist
  '(
    ("Symbols" "psy")
    ("ZapfDingbats" "pzd")
    ))

(defun LaTeX-pifont-display-font-table (family)
  (let ((pic (expand-file-name
	      (concat family ".gif")
	      LaTeX-pifont-font-table-directory))
	num)
    (if (file-exists-p pic)
	(progn
	  (let ((buf (get-buffer-create "*LaTeX-pifont*"))
		ext gl)
	    (save-window-excursion
	      (save-excursion
		(set-buffer buf)
		(setq ext (extent-at (point-min) (current-buffer) 'pifont-fam))
		(cond ((and ext (string= family (extent-property ext 'pifont-fam)))
		       ;; reuse it.
		       t)
		      (ext
		       ;; insert new glyph and property
		       (setq gl (make-glyph (vector 'gif :file pic)))
		       (set-extent-property ext 'begin-glyph gl)
		       (set-extent-property ext 'pifont-fam family))
		      (t
		       ;; fresh buffer: do everything
		       (setq ext (make-extent (point-min) (point-min)))
		       (set-extent-property ext 'end-closed t)
		       (setq gl (make-glyph (vector 'gif :file pic)))
		       (set-extent-property ext 'begin-glyph gl)
		       (set-extent-property ext 'pifont-fam family)
		       (setq buffer-read-only nil)
		       (insert " ")
		       (setq buffer-read-only t))
		      )
		(or (one-window-p)
		    (delete-other-windows))
		(set-window-buffer (selected-window) (current-buffer))
		(setq num (string-to-number (read-string
					     "Symbol number: "
					     nil nil "-1")))
		(bury-buffer)
	      ))
	    num))
      (error "font table for family %s doesn't exists" family))))

(defun LaTeX-pifont-cmd-ding (ignore)
  (let ((num (LaTeX-pifont-display-font-table "ZapfDingbats")))
    ;; FIXME: should also check if the symbol exists.
    (if (or (> 0 num) (< 255 num))
	(error "out of range symbol number %d" num)
      (insert (format "{%d}" num)))
    ))

(defun LaTeX-pifont-env-ding (env)
  (let ((num (LaTeX-pifont-display-font-table "ZapfDingbats")))
    ;; FIXME: should also check if the symbol exists.
    (if (or (> 0 num) (< 255 num))
	(error "out of range symbol number %d" num)
      (LaTeX-insert-environment
       env
       (concat TeX-grop (int-to-string num) TeX-grcl))
      (and (LaTeX-label environment)
	   (newline-and-indent))
      )
    ))

(defun LaTeX-pifont-cmd-pi (optional args)
  (let ((family (completing-read
		 "Font name: "
		 LaTeX-pifont-font-alist))
	num KBfam str)
    (if (or (string= "" family)
	    (not (assoc family LaTeX-pifont-font-alist)))
	(error "cannot use family %s" family)
      (setq KBfam (car (cdr (assoc family LaTeX-pifont-font-alist))))
      (cond ((= args 1)
	     (setq str (format "{%s}" KBfam)))
	    ((= args 2)
	     (setq num (LaTeX-pifont-display-font-table family))
	     ;; FIXME: should also check if the symbol exists.
	     (if (or (> 0 num) (< 255 num))
		 (error "out of range symbol number %d" num)
	      )
	     (setq str (format "{%s}{%d}" KBfam num)))
	    (t
	     (error "wrong number of arguments"))
	    )
      (insert str))
    ))

(defun LaTeX-pifont-env-pi (env)
  (let ((family (completing-read
		 "Font name: "
		 LaTeX-pifont-font-alist))
	num KBfam)
    (if (or (string= "" family)
	    (not (assoc family LaTeX-pifont-font-alist)))
	(error "cannot use family %s" family)
      (setq num (LaTeX-pifont-display-font-table family))
      ;; FIXME: should also check if the symbol exists.
      (if (or (> 0 num) (< 255 num))
	  (error "out of range symbol number %d" num)
	(setq KBfam (car (cdr (assoc family LaTeX-pifont-font-alist))))
	(LaTeX-insert-environment
	 env
	 (concat TeX-grop KBfam TeX-grcl
		 TeX-grop (int-to-string num) TeX-grcl))
	(and (LaTeX-label environment)
	     (newline-and-indent))
	))
    ))
