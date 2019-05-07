

vpath %.cc $(TARGET)

ROOT_PATH=../..
MKINC_PATH=$(ROOT_PATH)/makes
include $(MKINC_PATH)/defines.inc

SONAME    =libnvcaffe_parser.so
SODNAME   =libnvcaffe_parser_debug.so
LIBNAME   =libnvcaffe_parser.a
LIBDNAME  =libnvcaffe_parser_debug.a
PONAME    =libnvparsers.so
PODNAME   =libnvparsers_debug.so
ifeq ($(ABITYPE),androideabi)
SONAME_VERSIONED_MAJOR = $(SONAME)
SONAME_VERSIONED_MINOR = $(SONAME)
SODNAME_VERSIONED_MAJOR = $(SODNAME)
SODNAME_VERSIONED_MINOR = $(SODNAME)
PONAME_VERSIONED_MINOR = $(PONAME)
PODNAME_VERSIONED_MINOR = $(PODNAME)
else
SONAME_VERSIONED_MAJOR = $(SONAME).$(TRT_SO_MAJOR)
SONAME_VERSIONED_MINOR = $(SONAME).$(TRT_SO_MAJOR).$(TRT_SO_MINOR).$(TRT_SO_PATCH)
SODNAME_VERSIONED_MAJOR = $(SODNAME).$(TRT_SO_MAJOR)
SODNAME_VERSIONED_MINOR = $(SODNAME).$(TRT_SO_MAJOR).$(TRT_SO_MINOR).$(TRT_SO_PATCH)
PONAME_VERSIONED_MINOR = $(PONAME).$(TRT_SO_MAJOR).$(TRT_SO_MINOR).$(TRT_SO_PATCH)
PODNAME_VERSIONED_MINOR = $(PODNAME).$(TRT_SO_MAJOR).$(TRT_SO_MINOR).$(TRT_SO_PATCH)
endif

COMMON_FLAGS += -Dgoogle=google_private

CFLAGS    =$(COMMON_FLAGS)
CFLAGSD   =$(COMMON_FLAGS) -g
SOCFLAGS  =$(CFLAGS) -D_REENTRANT -fPIC
SOCFLAGSD =$(CFLAGSD) -D_REENTRANT -fPIC

LFLAGS  += $(COMMON_LD_FLAGS) -shared
LFLAGSD += $(COMMON_LD_FLAGS) -shared

.PHONY: install install_debug all release debug tidy
######################################################################### release, debug: lib

all:

release:

debug:

install:
	$(call check-install-dir)
	$(AT)ln -rnfs $(INSTALL_LIB_DIR)/libnvparsers_static.a $(INSTALL_LIB_DIR)/libnvcaffe_parser.a
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(PONAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SONAME_VERSIONED_MINOR))
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SONAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SONAME).$(TRT_SO_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SONAME).$(TRT_SO_MAJOR),$(INSTALL_LIB_DIR)/$(SONAME))
	$(AT)if [ ! "$(ABITYPE)" != "androideabi" ]; then ln -rnfs $(INSTALL_LIB_DIR)/$(PONAME_VERSIONED_MINOR) $(INSTALL_LIB_DIR)/$(SONAME); fi

install_debug:
	$(call check-install-dir)
	$(AT)ln -rfns $(INSTALL_LIB_DIR)/libnvparsers_static_debug.a $(INSTALL_LIB_DIR)/libnvcaffe_parser_debug.a
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(PODNAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SODNAME_VERSIONED_MINOR))
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SODNAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SODNAME).$(TRT_SO_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SODNAME).$(TRT_SO_MAJOR),$(INSTALL_LIB_DIR)/$(SODNAME))
	$(call symlink_safe_android,$(ABITYPE),$(INSTALL_LIB_DIR)/$(PODNAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SODNAME))

######################################################################### release static lib : objs
release:
	$(AT)if [ ! -d $(OUTDIR) ]; then mkdir -p $(OUTDIR); fi
	$(AT)ln -rnfs $(OUTDIR)/libnvparsers_static.a $(OUTDIR)/libnvcaffe_parser.a
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(PONAME_VERSIONED_MINOR),$(OUTDIR)/$(SONAME_VERSIONED_MINOR))
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SONAME_VERSIONED_MINOR),$(OUTDIR)/$(SONAME).$(TRT_SO_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SONAME).$(TRT_SO_MAJOR),$(OUTDIR)/$(SONAME))
	$(call symlink_safe_android,$(ABITYPE),$(OUTDIR)/$(PONAME_VERSIONED_MINOR),$(OUTDIR)/$(SONAME))

######################################################################### debug static lib : objs
debug:
	$(AT)if [ ! -d $(OUTDIR) ]; then mkdir -p $(OUTDIR); fi
	$(AT)ln -rnfs $(OUTDIR)/libnvparsers_static_debug.a $(OUTDIR)/libnvcaffe_parser_debug.a
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(PODNAME_VERSIONED_MINOR),$(OUTDIR)/$(SODNAME_VERSIONED_MINOR))
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SODNAME_VERSIONED_MINOR),$(OUTDIR)/$(SODNAME).$(TRT_SO_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SODNAME).$(TRT_SO_MAJOR),$(OUTDIR)/$(SODNAME))
	$(call symlink_safe_android,$(ABITYPE),$(OUTDIR)/$(PODNAME_VERSIONED_MINOR),$(OUTDIR)/$(SODNAME))

######################################################################### CLANG-TIDY support
include $(ROOT_PATH)/makes/tidy.inc

######################################################################### CLEAN

clean:

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d $(DOBJDIR)/*.d $(SOOBJDIR)/*.d $(SODOBJDIR)/*.d
endif

